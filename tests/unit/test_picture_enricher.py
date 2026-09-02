"""Unit tests for picture_enricher (Phase 11).

The LLM is stubbed at the RoutedLLM boundary: these check routing, batching, what reaches the
model, what lands on pic.meta.description, and that every failure mode degrades instead of
raising. The adapter-level image serialization is covered by test_openai_adapter.py.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Any

import pytest
from docling_core.types.doc.document import (
    DoclingDocument,
    ImageRef,
    PictureClassificationMetaField,
    PictureClassificationPrediction,
    PictureItem,
    PictureMeta,
)
from PIL import Image

from src.services.ingestion import picture_enricher


def _png(color: str = "red") -> Image.Image:
    return Image.new("RGB", (8, 8), color)


def _doc(*specs: tuple[str, float | None] | None) -> DoclingDocument:
    """Build a document with one picture per spec: (label, confidence), or None for no
    classification at all. Every picture gets a real in-memory crop."""
    doc = DoclingDocument(name="test")
    for i, spec in enumerate(specs):
        meta = None
        if spec is not None:
            label, confidence = spec
            meta = PictureMeta(
                classification=PictureClassificationMetaField(
                    predictions=[
                        PictureClassificationPrediction(class_name=label, confidence=confidence)
                    ]
                )
            )
        pic = PictureItem(
            self_ref=f"#/pictures/{i}",
            image=ImageRef.from_pil(_png(), dpi=72),
            meta=meta,
            prov=[],
        )
        doc.pictures.append(pic)
    return doc


@dataclass
class _Resp:
    """Minimal stand-in for LLMResponse: the enricher only reads .text."""

    text: str


class _StubLLM:
    """Stands in for RoutedLLM, recording calls and replying with a canned description."""

    def __init__(self, *, vision: bool = True, text: str | None = None) -> None:
        self.capabilities: dict[str, Any] = {"vision": vision}
        self.calls: list[dict[str, Any]] = []
        self._text = text

    async def complete(self, messages, **params):  # noqa: ANN001, ANN003
        self.calls.append({"messages": messages, "params": params})
        images = messages[-1].images or ()
        body = self._text
        if body is None:
            body = json.dumps(
                {
                    "descriptions": [
                        {"picture_id": i, "description": f"desc {i}"}
                        for i in range(1, len(images) + 1)
                    ]
                }
            )
        return _Resp(body)


class _RaisingLLM(_StubLLM):
    async def complete(self, messages, **params):  # noqa: ANN001, ANN003
        self.calls.append({"messages": messages, "params": params})
        raise RuntimeError("provider exploded")


@pytest.fixture
def lanes(monkeypatch: pytest.MonkeyPatch) -> dict[str, _StubLLM]:
    """Install stub LLMs for both lanes and bypass the router/prompt loader."""
    chart, caption = _StubLLM(), _StubLLM()
    _install(monkeypatch, chart, caption)
    return {"chart": chart, "caption": caption}


def _install(monkeypatch: pytest.MonkeyPatch, chart: Any, caption: Any) -> None:
    picture_enricher.reset()
    monkeypatch.setattr(picture_enricher, "_ensure_initialized", lambda: None)
    monkeypatch.setattr(picture_enricher, "_chart_llm", chart)
    monkeypatch.setattr(picture_enricher, "_caption_llm", caption)
    monkeypatch.setattr(picture_enricher, "_chart_prompt", "chart prompt")
    monkeypatch.setattr(picture_enricher, "_caption_prompt", "caption prompt")
    monkeypatch.setattr(picture_enricher, "_chart_model", "chart-model")
    monkeypatch.setattr(picture_enricher, "_caption_model", "caption-model")


def _descriptions(doc: DoclingDocument) -> list[str | None]:
    return [
        pic.meta.description.text if pic.meta and pic.meta.description else None
        for pic in doc.pictures
    ]


class TestRouting:
    async def test_charts_and_captions_go_to_different_models(
        self, lanes: dict[str, _StubLLM]
    ) -> None:
        doc = _doc(("bar_chart", 0.9), ("screenshot", 0.9))
        assert await picture_enricher.enrich_pictures(doc) == 2
        assert len(lanes["chart"].calls) == 1
        assert len(lanes["caption"].calls) == 1

    @pytest.mark.parametrize("label", sorted(picture_enricher.SKIP_LABELS))
    async def test_skip_labels_cost_nothing(self, lanes: dict[str, _StubLLM], label: str) -> None:
        """A described signature or logo is noise that gets embedded as document text."""
        doc = _doc((label, 0.99))
        assert await picture_enricher.enrich_pictures(doc) == 0
        assert lanes["chart"].calls == [] and lanes["caption"].calls == []
        assert _descriptions(doc) == [None]

    @pytest.mark.parametrize("label", sorted(picture_enricher.CHART_LABELS))
    async def test_chart_labels_take_the_strong_lane(
        self, lanes: dict[str, _StubLLM], label: str
    ) -> None:
        doc = _doc((label, 0.9))
        await picture_enricher.enrich_pictures(doc)
        assert len(lanes["chart"].calls) == 1

    async def test_unknown_label_falls_through_to_caption(self, lanes: dict[str, _StubLLM]) -> None:
        """A label from a future classifier revision degrades to a cheap caption, not silence."""
        doc = _doc(("newly_invented_class", 0.99))
        assert await picture_enricher.enrich_pictures(doc) == 1
        assert len(lanes["caption"].calls) == 1

    async def test_missing_classification_is_captioned(self, lanes: dict[str, _StubLLM]) -> None:
        doc = _doc(None)
        assert await picture_enricher.enrich_pictures(doc) == 1
        assert len(lanes["caption"].calls) == 1

    async def test_low_confidence_chart_is_demoted_to_caption(
        self, lanes: dict[str, _StubLLM], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Trusting a 0.2-confidence bar_chart would spend a high-detail call on a guess."""
        monkeypatch.setenv("PICTURE_ENRICHER_MIN_CONFIDENCE", "0.5")
        doc = _doc(("bar_chart", 0.2))
        await picture_enricher.enrich_pictures(doc)
        assert lanes["chart"].calls == []
        assert len(lanes["caption"].calls) == 1

    async def test_picture_without_a_crop_is_skipped(self, lanes: dict[str, _StubLLM]) -> None:
        doc = _doc(("bar_chart", 0.9))
        doc.pictures[0].image = None
        assert await picture_enricher.enrich_pictures(doc) == 0
        assert lanes["chart"].calls == []


class TestRequestShape:
    async def test_detail_is_high_for_charts_and_low_for_captions(
        self, lanes: dict[str, _StubLLM]
    ) -> None:
        """The cost lever: reading axis labels needs high, a one-line caption does not."""
        doc = _doc(("line_chart", 0.9), ("map", 0.9))
        await picture_enricher.enrich_pictures(doc)
        chart_img = lanes["chart"].calls[0]["messages"][-1].images[0]
        caption_img = lanes["caption"].calls[0]["messages"][-1].images[0]
        assert chart_img.detail == "high"
        assert caption_img.detail == "low"
        assert chart_img.mime_type == "image/png"

    async def test_label_is_written_into_the_text_part(self, lanes: dict[str, _StubLLM]) -> None:
        """The model sees pixels and this text only — the classifier label reaches it no
        other way."""
        doc = _doc(("pie_chart", 0.9))
        await picture_enricher.enrich_pictures(doc)
        assert lanes["chart"].calls[0]["messages"][-1].content == "[PICTURE 1] pie_chart"

    async def test_system_prompt_and_structured_output_are_set(
        self, lanes: dict[str, _StubLLM]
    ) -> None:
        doc = _doc(("bar_chart", 0.9))
        await picture_enricher.enrich_pictures(doc)
        call = lanes["chart"].calls[0]
        assert call["messages"][0].content == "chart prompt"
        assert call["params"]["response_format"]["json_schema"]["name"] == "picture_descriptions"
        assert call["params"]["temperature"] == 0.0

    async def test_batching_splits_calls_and_numbers_restart_per_batch(
        self, lanes: dict[str, _StubLLM], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PICTURE_ENRICHER_BATCH_SIZE", "2")
        doc = _doc(*[("bar_chart", 0.9)] * 5)
        assert await picture_enricher.enrich_pictures(doc) == 5
        calls = lanes["chart"].calls
        assert [len(c["messages"][-1].images) for c in calls] == [2, 2, 1]
        assert calls[-1]["messages"][-1].content == "[PICTURE 1] bar_chart"


class TestResultWriting:
    @pytest.mark.usefixtures("lanes")
    async def test_description_and_model_land_on_meta(self) -> None:
        doc = _doc(("bar_chart", 0.9))
        await picture_enricher.enrich_pictures(doc)
        meta = doc.pictures[0].meta
        assert meta is not None and meta.description is not None
        assert meta.description.text == "desc 1"
        assert meta.description.created_by == "chart-model"  # provenance for the artifact

    @pytest.mark.usefixtures("lanes")
    async def test_meta_is_created_when_absent(self) -> None:
        doc = _doc(None)
        assert doc.pictures[0].meta is None
        await picture_enricher.enrich_pictures(doc)
        meta = doc.pictures[0].meta
        assert meta is not None and meta.description is not None
        assert meta.description.text == "desc 1"

    async def test_empty_description_leaves_meta_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The prompt's escape hatch for an unreadable image. Unset is what the chunker drops,
        and it keeps created_by off the artifact so a skip is distinguishable from a describe."""
        body = json.dumps({"descriptions": [{"picture_id": 1, "description": "   "}]})
        _install(monkeypatch, _StubLLM(text=body), _StubLLM(text=body))
        doc = _doc(("bar_chart", 0.9))
        assert await picture_enricher.enrich_pictures(doc) == 0
        assert _descriptions(doc) == [None]

    async def test_partial_response_writes_only_what_came_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        body = json.dumps({"descriptions": [{"picture_id": 2, "description": "only the second"}]})
        _install(monkeypatch, _StubLLM(text=body), _StubLLM(text=body))
        doc = _doc(("bar_chart", 0.9), ("bar_chart", 0.9))
        assert await picture_enricher.enrich_pictures(doc) == 1
        assert _descriptions(doc) == [None, "only the second"]

    async def test_out_of_range_id_is_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A hallucinated picture_id must not write onto the wrong picture."""
        body = json.dumps({"descriptions": [{"picture_id": 99, "description": "nowhere"}]})
        _install(monkeypatch, _StubLLM(text=body), _StubLLM(text=body))
        doc = _doc(("bar_chart", 0.9))
        assert await picture_enricher.enrich_pictures(doc) == 0
        assert _descriptions(doc) == [None]


class _FlakyLLM(_StubLLM):
    """Fails the first `fail_times` attempts, then answers normally. Models the observed
    failure: HTTP 200 with an empty body on the chart lane."""

    def __init__(self, *, fail_times: int = 1, body: str = "") -> None:
        super().__init__()
        self._fail_times = fail_times
        self._body = body

    async def complete(self, messages, **params):  # noqa: ANN001, ANN003
        if len(self.calls) < self._fail_times:
            self.calls.append({"messages": messages, "params": params})
            return _Resp(self._body)
        return await super().complete(messages, **params)


class TestRetry:
    async def test_empty_body_is_retried_and_recovers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The first real smoke run hit exactly this: same image, 200, empty body."""
        flaky = _FlakyLLM(fail_times=1)
        _install(monkeypatch, flaky, _StubLLM())
        doc = _doc(("pie_chart", 0.9))
        assert await picture_enricher.enrich_pictures(doc) == 1
        assert len(flaky.calls) == 2
        assert _descriptions(doc) == ["desc 1"]

    async def test_malformed_json_is_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        flaky = _FlakyLLM(fail_times=1, body="here is your description!")
        _install(monkeypatch, flaky, _StubLLM())
        doc = _doc(("pie_chart", 0.9))
        assert await picture_enricher.enrich_pictures(doc) == 1
        assert len(flaky.calls) == 2

    async def test_retry_is_capped_at_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        flaky = _FlakyLLM(fail_times=99)
        _install(monkeypatch, flaky, _StubLLM())
        doc = _doc(("pie_chart", 0.9))
        assert await picture_enricher.enrich_pictures(doc) == 0
        assert len(flaky.calls) == 2  # not an unbounded loop on a persistent failure

    async def test_escape_hatch_is_not_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A parsed response with an empty description is the prompt's "unreadable image"
        answer. Retrying it pays twice for the same verdict."""
        body = json.dumps({"descriptions": [{"picture_id": 1, "description": ""}]})
        stub = _StubLLM(text=body)
        _install(monkeypatch, stub, _StubLLM())
        doc = _doc(("pie_chart", 0.9))
        assert await picture_enricher.enrich_pictures(doc) == 0
        assert len(stub.calls) == 1
        assert _descriptions(doc) == [None]

    async def test_exception_is_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        raising = _RaisingLLM()
        _install(monkeypatch, raising, _StubLLM())
        doc = _doc(("pie_chart", 0.9))
        assert await picture_enricher.enrich_pictures(doc) == 0
        assert len(raising.calls) == 2


class TestTokenBudget:
    async def test_budget_scales_with_batch_size(self, lanes: dict[str, _StubLLM]) -> None:
        """1200/picture, not 400: on GPT-5 models this budget also covers reasoning tokens."""
        doc = _doc(("bar_chart", 0.9), ("bar_chart", 0.9))
        await picture_enricher.enrich_pictures(doc)
        assert lanes["chart"].calls[0]["params"]["max_tokens"] == 2400


class TestDegradation:
    """The contract: enrichment degrades to Phase 4 quality, never fails a document."""

    async def test_batch_failure_is_isolated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install(monkeypatch, _RaisingLLM(), _StubLLM())
        doc = _doc(("bar_chart", 0.9), ("screenshot", 0.9))
        assert await picture_enricher.enrich_pictures(doc) == 1  # caption lane still ran
        assert _descriptions(doc) == [None, "desc 1"]

    async def test_unparseable_response_degrades(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install(monkeypatch, _StubLLM(text="not json at all"), _StubLLM(text="not json"))
        doc = _doc(("bar_chart", 0.9))
        assert await picture_enricher.enrich_pictures(doc) == 0
        assert _descriptions(doc) == [None]

    async def test_document_with_no_pictures_makes_no_calls(
        self, lanes: dict[str, _StubLLM]
    ) -> None:
        assert await picture_enricher.enrich_pictures(_doc()) == 0
        assert lanes["chart"].calls == [] and lanes["caption"].calls == []


class TestVisionCapabilityGuard:
    async def test_text_only_model_is_rejected_at_init(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fail at startup, not mid-ingestion on a paid call."""
        picture_enricher.reset()

        class _Router:
            def get(self, _model: str) -> Any:
                return _StubLLM(vision=False)

        monkeypatch.setattr(picture_enricher, "get_router", lambda: _Router())
        with pytest.raises(ValueError, match="not marked vision-capable"):
            picture_enricher._ensure_initialized()
        picture_enricher.reset()


class TestCropEncoding:
    def test_crop_bytes_are_png(self) -> None:
        doc = _doc(("bar_chart", 0.9))
        data = picture_enricher._crop_bytes(doc.pictures[0])
        assert data is not None
        assert Image.open(io.BytesIO(data)).format == "PNG"
