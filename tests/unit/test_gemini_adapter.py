"""Unit tests for the Gemini adapter's request construction (Phase 10: multimodal support).

These mock `client.aio.models.generate_content` rather than the HTTP layer. respx cannot
reach this adapter: google-genai uses aiohttp for its async path whenever aiohttp is
installed (it is, via the SDK's own extras), and respx only patches httpx. The seam below
still exercises the whole adapter — role mapping, part construction, media_resolution and
response parsing — everything except the SDK's own serialization of the objects we build.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest
from google.genai import types

from src.services.llm_adapters.base_adapter import ChatMessage, ImagePart, Role
from src.services.llm_adapters.gemini_adapter import GeminiAdapter

PNG = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def adapter(monkeypatch: pytest.MonkeyPatch) -> GeminiAdapter:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    return GeminiAdapter(default_model="gemini-3.7-flash", api_key="test-key")


class _Capture:
    """Stands in for generate_content, recording the request the adapter built."""

    def __init__(self) -> None:
        self.contents: Any = None
        self.config: Any = None

    async def __call__(self, *, model: str, contents: Any, config: Any) -> Any:  # noqa: ARG002
        self.contents, self.config = contents, config
        return SimpleNamespace(
            text="ok",
            usage_metadata=SimpleNamespace(
                prompt_token_count=5, candidates_token_count=1, total_token_count=6
            ),
        )


async def _capture(adapter: GeminiAdapter, messages: list[ChatMessage], model: str) -> _Capture:
    cap = _Capture()
    adapter._client.aio.models.generate_content = cap  # type: ignore[method-assign]
    resp = await adapter.complete(messages, model=model)
    assert resp.text == "ok"  # the response path still runs
    return cap


def _image_parts(content: types.Content) -> list[types.Part]:
    return [p for p in (content.parts or []) if p.inline_data is not None]


def _levels(content: types.Content) -> list[str | None]:
    out: list[str | None] = []
    for p in _image_parts(content):
        mr = p.media_resolution
        out.append(mr.level.value if mr is not None and mr.level is not None else None)
    return out


class TestTextOnlyUnchanged:
    async def test_roles_and_system_instruction(self, adapter: GeminiAdapter) -> None:
        cap = await _capture(
            adapter,
            [
                ChatMessage(role=Role.system, content="sys"),
                ChatMessage(role=Role.user, content="hi"),
                ChatMessage(role=Role.assistant, content="prior"),
            ],
            "gemini-3.7-flash",
        )
        assert cap.config.system_instruction == "sys"
        assert [c.role for c in cap.contents] == ["user", "model"]
        assert [c.parts[0].text for c in cap.contents] == ["hi", "prior"]
        assert cap.config.media_resolution is None


class TestImageParts:
    async def test_text_then_image_on_one_content(self, adapter: GeminiAdapter) -> None:
        cap = await _capture(
            adapter,
            [
                ChatMessage(
                    role=Role.user, content="describe", images=(ImagePart(PNG, "image/png"),)
                )
            ],
            "gemini-3.7-flash",
        )
        parts = cap.contents[0].parts
        assert parts[0].text == "describe"
        assert parts[1].inline_data.data == PNG
        assert parts[1].inline_data.mime_type == "image/png"

    async def test_image_only_message_omits_the_text_part(self, adapter: GeminiAdapter) -> None:
        cap = await _capture(
            adapter,
            [ChatMessage(role=Role.user, content=None, images=(ImagePart(PNG, "image/png"),))],
            "gemini-3.7-flash",
        )
        assert len(cap.contents[0].parts) == 1
        assert cap.contents[0].parts[0].inline_data is not None

    async def test_images_on_system_role_raise(self, adapter: GeminiAdapter) -> None:
        """system/developer collapse into system_instruction as text, so an image there
        would vanish without a trace."""
        with pytest.raises(ValueError, match="not supported on system messages"):
            await _capture(
                adapter,
                [ChatMessage(role=Role.system, content="s", images=(ImagePart(PNG, "image/png"),))],
                "gemini-3.7-flash",
            )


class TestPerPartResolutionOnGemini3:
    @pytest.mark.parametrize(
        ("detail", "expected"),
        [("low", "MEDIA_RESOLUTION_LOW"), ("high", "MEDIA_RESOLUTION_HIGH"), ("auto", None)],
    )
    async def test_detail_maps_to_the_part(
        self, adapter: GeminiAdapter, detail: str, expected: str | None
    ) -> None:
        cap = await _capture(
            adapter,
            [
                ChatMessage(
                    role=Role.user,
                    content="x",
                    images=(ImagePart(PNG, "image/png", detail=detail),),  # type: ignore[arg-type]
                )
            ],
            "gemini-3.7-flash",
        )
        assert _levels(cap.contents[0]) == [expected]
        assert cap.config.media_resolution is None  # never both levels at once

    async def test_parts_keep_independent_levels(self, adapter: GeminiAdapter) -> None:
        """The whole point of the per-part API: a chart at high beside a caption at low."""
        cap = await _capture(
            adapter,
            [
                ChatMessage(
                    role=Role.user,
                    content="x",
                    images=(
                        ImagePart(PNG, "image/png", "high"),
                        ImagePart(PNG, "image/png", "low"),
                    ),
                )
            ],
            "gemini-3.7-flash",
        )
        assert _levels(cap.contents[0]) == ["MEDIA_RESOLUTION_HIGH", "MEDIA_RESOLUTION_LOW"]


class TestRequestLevelFallback:
    """No configured model reaches this branch today (models.yaml has only gemini-3.7-flash),
    so these tests are the only thing holding the merge rule in place."""

    MODEL = "gemini-2.5-flash"

    async def test_max_detail_wins(self, adapter: GeminiAdapter) -> None:
        """Under-resolving a chart yields a confident unreadable description; over-resolving
        a caption just costs tokens. The asymmetry is why it is max and not first or last."""
        cap = await _capture(
            adapter,
            [
                ChatMessage(
                    role=Role.user,
                    content="x",
                    images=(
                        ImagePart(PNG, "image/png", "low"),
                        ImagePart(PNG, "image/png", "high"),
                    ),
                )
            ],
            self.MODEL,
        )
        assert cap.config.media_resolution == types.MediaResolution.MEDIA_RESOLUTION_HIGH
        assert _levels(cap.contents[0]) == [None, None]  # per-part is Gemini 3 only

    async def test_all_auto_sets_nothing_and_stays_quiet(
        self, adapter: GeminiAdapter, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            cap = await _capture(
                adapter,
                [
                    ChatMessage(
                        role=Role.user,
                        content="x",
                        images=(ImagePart(PNG, "image/png"), ImagePart(PNG, "image/png")),
                    )
                ],
                self.MODEL,
            )
        assert cap.config.media_resolution is None
        assert "media_resolution_merged" not in caplog.text

    async def test_merge_is_logged(
        self, adapter: GeminiAdapter, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A silently ignored `high` means paying for a description that could not read the
        chart, so the drop has to be visible in the logs."""
        with caplog.at_level(logging.WARNING):
            await _capture(
                adapter,
                [
                    ChatMessage(
                        role=Role.user,
                        content="x",
                        images=(
                            ImagePart(PNG, "image/png", "low"),
                            ImagePart(PNG, "image/png", "high"),
                        ),
                    )
                ],
                self.MODEL,
            )
        record = next(
            r for r in caplog.records if r.message == "gemini_adapter.media_resolution_merged"
        )
        assert record.model == self.MODEL  # type: ignore[attr-defined]
        assert record.requested == ["high", "low"]  # type: ignore[attr-defined]
        assert record.applied == "high"  # type: ignore[attr-defined]


class TestUnsupportedSurfaces:
    async def test_stream_rejects_images(self, adapter: GeminiAdapter) -> None:
        """GeminiAdapter does not override stream(), so this is the base-class guard firing."""
        msgs = [ChatMessage(role=Role.user, content="x", images=(ImagePart(PNG, "image/png"),))]
        with pytest.raises(ValueError, match="does not support image content"):
            adapter.stream(msgs)
