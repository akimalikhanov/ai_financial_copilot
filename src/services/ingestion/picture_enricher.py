"""Describe a document's pictures with a vision model, routed by classification label.

Producer side of the picture description contract: this writes `pic.meta.description`, which
`chunker._build_picture_descriptions` reads to replace `<!-- image -->` placeholders with real
text. It therefore has to run *before* `chunk_document`, and before the artifacts are exported
so the persisted JSON carries the descriptions.

Modelled on `table_summarizer.py`: batched LLM calls through the router, structured output, and
per-batch failure isolation. A failure here leaves `pic.meta.description` unset, which is the
state the chunker already drops — enrichment degrades to Phase 4 quality, never fails a document.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Any

from src.observability.metrics import PICTURE_ENRICHER, PICTURE_ENRICHER_DURATION
from src.schemas.picture_enricher import PictureDescriptionResponse
from src.services.llm_adapters.base_adapter import ChatMessage, ImagePart, Role
from src.services.llm_router import RoutedLLM, get_router
from src.services.prompts.prompt_loader import get_prompt_loader
from src.utils.config import (
    get_picture_enricher_batch_size,
    get_picture_enricher_cheap_model,
    get_picture_enricher_concurrency,
    get_picture_enricher_min_completion_tokens,
    get_picture_enricher_min_confidence,
    get_picture_enricher_model,
    get_picture_enricher_reasoning_effort,
    get_picture_enricher_seconds_per_picture,
    get_picture_enricher_stage_timeout,
)
from src.utils.json_schema import build_response_format

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument, PictureItem

logger = logging.getLogger(__name__)

# Lanes, keyed by the 16 labels docling's document figure classifier actually emits. "other" is
# the catch-all the classifier assigns when a picture doesn't fit any of its named classes —
# financial-report stat-tile/KPI infographic panels land here, and they are exactly as
# numbers-dense as a chart, so they get the chart lane's high-detail image and number-preserving
# prompt rather than the caption lane's one-sentence, no-numbers treatment. Anything else unlisted
# (including a label from a future model revision) still falls through to CAPTION, so a genuinely
# new class degrades to a cheap caption rather than to silence.
CHART_LABELS = frozenset({"bar_chart", "line_chart", "pie_chart", "flow_chart", "other"})
# No model call: these carry no retrievable content, and a description of a signature or a
# decorative icon is noise that gets embedded and BM25-indexed as document text.
SKIP_LABELS = frozenset({"logo", "icon", "signature", "stamp", "qr_code", "bar_code"})

_MIME = "image/png"  # what _crop_bytes writes; same format as tasks._encode_picture_crop
# One retry: the chart lane occasionally returns HTTP 200 with an empty body.
_MAX_ATTEMPTS = 2
# Completion budget per picture, floored by PICTURE_ENRICHER_MIN_COMPLETION_TOKENS so that a
# short tail batch is not starved.
_TOKENS_PER_PICTURE = 1200


@dataclass(frozen=True, slots=True)
class _Candidate:
    """One picture to describe and the lane it routed to.

    Holds no crop bytes: those are encoded per batch inside `_run_lane`, under the semaphore,
    so at most `concurrency x batch_size` PNGs are live rather than one per picture.
    """

    picture: PictureItem
    label: str


@dataclass(frozen=True, slots=True)
class _Encoded:
    """A candidate whose crop has been encoded for the call in flight."""

    candidate: _Candidate
    image: bytes


_chart_prompt: str | None = None
_caption_prompt: str | None = None
_chart_llm: RoutedLLM | None = None
_caption_llm: RoutedLLM | None = None
_chart_model: str | None = None
_caption_model: str | None = None


def _ensure_initialized() -> None:
    """Lazy-init on first use (after fork reset clears stale state)."""
    global _chart_prompt, _caption_prompt, _chart_llm, _caption_llm, _chart_model, _caption_model
    if _chart_llm is not None:
        return
    loader = get_prompt_loader()
    _chart_prompt = loader.load("picture_enricher_chart", "v1").template
    _caption_prompt = loader.load("picture_enricher_caption", "v1").template
    _chart_model = get_picture_enricher_model()
    _caption_model = get_picture_enricher_cheap_model()
    router = get_router()
    _chart_llm = router.get(_chart_model)
    _caption_llm = router.get(_caption_model)
    for llm, model in ((_chart_llm, _chart_model), (_caption_llm, _caption_model)):
        if not llm.capabilities.get("vision", False):
            raise ValueError(
                f"picture enricher model {model!r} is not marked vision-capable in models.yaml"
            )


def validate_config() -> None:
    """Resolve models and prompts now, raising on a misconfiguration.

    Called at worker start so a bad PICTURE_ENRICHER_MODEL fails the pod loudly instead of
    degrading every document forever behind the call site's bare `except Exception`.
    """
    _ensure_initialized()


def reset() -> None:
    """Clear cached state (call after fork)."""
    global _chart_prompt, _caption_prompt, _chart_llm, _caption_llm, _chart_model, _caption_model
    _chart_prompt = _caption_prompt = None
    _chart_llm = _caption_llm = None
    _chart_model = _caption_model = None


# -- Routing ------------------------------------------------------------------


def _label_of(pic: PictureItem, min_confidence: float) -> str:
    """The picture's classification label, or "other" when absent or below the floor.

    A low-confidence label is worse than no label: it would route a misread chart to the
    caption lane and a misread logo to a paid chart call.
    """
    meta = getattr(pic, "meta", None)
    classification = getattr(meta, "classification", None) if meta is not None else None
    if classification is None:
        return "other"
    try:
        main = classification.get_main_prediction()
    except (ValueError, IndexError):
        return "other"
    confidence = main.confidence
    if confidence is not None and confidence < min_confidence:
        return "other"
    return main.class_name


def _crop_bytes(pic: PictureItem) -> bytes | None:
    """PNG bytes for the picture's in-memory crop, or None when it has none.

    Same source as tasks._encode_picture_crop: the crops exist on the document only because
    generate_picture_images=True, and only until tasks._upload_picture_crops clears them.
    """
    if pic.image is None:
        return None
    pil_image = pic.image.pil_image
    if pil_image is None:
        return None
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return buf.getvalue()


def _plan(
    document: DoclingDocument, min_confidence: float
) -> tuple[list[_Candidate], list[_Candidate]]:
    """Split the document's pictures into the chart lane and the caption lane, dropping the
    skip labels and anything with no crop to send. Encoding is deferred to `_encode_batch`."""
    charts: list[_Candidate] = []
    captions: list[_Candidate] = []
    for pic in document.pictures:
        label = _label_of(pic, min_confidence)
        if label in SKIP_LABELS or pic.image is None:
            continue
        candidate = _Candidate(picture=pic, label=label)
        (charts if label in CHART_LABELS else captions).append(candidate)
    return charts, captions


def _encode_batch(batch: list[_Candidate]) -> list[_Encoded]:
    """Encode a batch's crops, dropping any that fail to decode or come back empty."""
    encoded: list[_Encoded] = []
    for candidate in batch:
        try:
            image = _crop_bytes(candidate.picture)
        except Exception:
            logger.warning(
                "picture_enricher.crop_failed",
                extra={"self_ref": candidate.picture.self_ref},
                exc_info=True,
            )
            continue
        if image:
            encoded.append(_Encoded(candidate=candidate, image=image))
    return encoded


# -- LLM calls ----------------------------------------------------------------


def _build_message(batch: list[_Encoded], detail: str) -> ChatMessage:
    """One user message holding the whole batch: a text part naming each picture, then the
    crops in the same order. The model sees only pixels and this text, so the label has to be
    written in explicitly (see docs/notes/llm-image-inputs.md)."""
    lines = [f"[PICTURE {i}] {c.candidate.label}" for i, c in enumerate(batch, start=1)]
    return ChatMessage(
        role=Role.user,
        content="\n".join(lines),
        images=tuple(ImagePart(data=c.image, mime_type=_MIME, detail=detail) for c in batch),  # type: ignore[arg-type]
    )


def _parse(raw: str, size: int, lane: str = "unknown") -> dict[int, str] | None:
    """Parse structured output into {1-based picture id: description}, dropping blanks.

    Returns None when the response could not be parsed at all — an empty body or malformed
    JSON, both of which are model failures worth one retry. A parsed response with empty
    descriptions returns an empty mapping instead: that is the prompt's escape hatch for an
    unreadable image, and retrying it would just pay twice for the same answer.
    """
    if not raw.strip():
        # The signature of a starved completion budget: HTTP 200, no body, because reasoning
        # spent the whole allowance. Counted so 6.1's floor can be shown to have worked.
        PICTURE_ENRICHER.labels(lane, "empty").inc(size)
        logger.warning("picture_enricher.empty_response", extra={"lane": lane})
        return None
    try:
        parsed = PictureDescriptionResponse.model_validate(json.loads(raw))
    except Exception:
        logger.warning(
            "picture_enricher.batch_parse_failed", extra={"raw": raw[:500]}, exc_info=True
        )
        return None
    out: dict[int, str] = {}
    for item in parsed.descriptions:
        text = item.description.strip()
        # Leaving meta.description unset is what the chunker drops; writing "" would too, but
        # unset also keeps created_by off the artifact, which is how a skipped picture is told
        # apart from a described one.
        if text and 1 <= item.picture_id <= size:
            out[item.picture_id] = text
    if len(out) < len(parsed.descriptions):
        logger.info(
            "picture_enricher.descriptions_dropped",
            extra={"returned": len(parsed.descriptions), "kept": len(out)},
        )
    return out


async def _describe_batch(
    batch: list[_Encoded],
    *,
    llm: RoutedLLM,
    model: str,
    system_prompt: str,
    detail: str,
    lane: str,
    response_format: dict[str, Any],
) -> dict[int, str] | None:
    """One batch, with a single retry. None means every attempt failed.

    The retry exists because the chart lane occasionally returns HTTP 200 with an empty body.
    Without it that picture silently goes undescribed and looks indistinguishable from one the
    model declined to describe.
    """
    messages = [
        ChatMessage(role=Role.system, content=system_prompt),
        _build_message(batch, detail),
    ]
    # On GPT-5 models max_tokens maps to max_completion_tokens, which counts reasoning tokens
    # too, and reasoning does not shrink with the batch — hence the floor, and the low effort.
    max_tokens = max(get_picture_enricher_min_completion_tokens(), _TOKENS_PER_PICTURE * len(batch))
    reasoning_effort = get_picture_enricher_reasoning_effort()
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        started = perf_counter()
        try:
            # temperature is not dead code: the adapter discards it only for GPT-5 models
            # (openai_adapter._is_gpt_5_model). The chart lane ignores it; the caption lane —
            # gpt-4o-mini by default — genuinely runs at 0.0. Both lanes share this call site.
            resp = await llm.complete(
                messages,
                _lf_name=f"picture_enricher.{lane}",
                temperature=0.0,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                response_format=response_format,
            )
        except Exception:
            logger.warning(
                "picture_enricher.batch_call_failed",
                extra={"lane": lane, "model": model, "attempt": attempt},
                exc_info=True,
            )
        else:
            descriptions = _parse(resp.text, len(batch), lane)
            if descriptions is not None:
                return descriptions
        finally:
            PICTURE_ENRICHER_DURATION.labels(lane).observe(perf_counter() - started)
        if attempt < _MAX_ATTEMPTS:
            logger.info(
                "picture_enricher.batch_retry",
                extra={"lane": lane, "model": model, "batch_size": len(batch)},
            )
    return None


async def _run_lane(
    candidates: list[_Candidate],
    *,
    llm: RoutedLLM,
    model: str,
    system_prompt: str,
    detail: str,
    batch_size: int,
    lane: str,
    sem: asyncio.Semaphore,
) -> int:
    """Describe one lane's pictures in batches, writing meta.description in place.

    Batches run concurrently under `sem` — they are network waits, so overlapping them costs
    nothing but the semaphore. The semaphore is passed in rather than created here so both
    lanes share one budget: the provider's rate limit is per-account, not per-lane.
    """
    if not candidates:
        return 0

    from docling_core.types.doc.document import DescriptionMetaField, PictureMeta

    response_format = build_response_format(
        "picture_descriptions", PictureDescriptionResponse.model_json_schema()
    )

    batches = [candidates[i : i + batch_size] for i in range(0, len(candidates), batch_size)]

    def _write(start: int, batch: list[_Candidate], descriptions: dict[int, str] | None) -> int:
        if descriptions is None:
            # Per-batch isolation: the contract is that enrichment degrades, never fails.
            PICTURE_ENRICHER.labels(lane, "failed").inc(len(batch))
            logger.warning(
                "picture_enricher.batch_failed",
                extra={
                    "lane": lane,
                    "batch_start": start,
                    "batch_size": len(batch),
                    "model": model,
                },
            )
            return 0
        described = 0
        for local_id, candidate in enumerate(batch, start=1):
            text = descriptions.get(local_id)
            if not text:
                continue
            pic = candidate.picture
            if pic.meta is None:
                pic.meta = PictureMeta()
            pic.meta.description = DescriptionMetaField(text=text, created_by=model)
            described += 1
        PICTURE_ENRICHER.labels(lane, "described").inc(described)
        # Parsed fine but the model declined to describe these — its escape hatch for an
        # unreadable image, which is a different outcome from a failed call.
        PICTURE_ENRICHER.labels(lane, "skipped").inc(len(batch) - described)
        return described

    async def _one(start: int, batch: list[_Candidate]) -> int:
        """Describe one batch and write its descriptions before returning.

        Writing here rather than after the gather is what keeps completed batches when the
        stage budget cancels the rest. The crops are encoded under the semaphore and released
        on return; the 1-based ids in `descriptions` line up with the encoded subset.
        """
        async with sem:
            encoded = _encode_batch(batch)
            if not encoded:
                return 0
            descriptions = await _describe_batch(
                encoded,
                llm=llm,
                model=model,
                system_prompt=system_prompt,
                detail=detail,
                lane=lane,
                response_format=response_format,
            )
        return _write(start, [e.candidate for e in encoded], descriptions)

    # return_exceptions=True is what preserves "degrade, never fail" under gather: without it
    # one raising batch cancels its siblings and takes down the whole lane.
    results = await asyncio.gather(
        *(_one(i * batch_size, b) for i, b in enumerate(batches)),
        return_exceptions=True,
    )

    written = 0
    for result in results:
        if isinstance(result, BaseException):
            # _describe_batch swallows call failures itself, so reaching here means something
            # unexpected — a cancellation, or a bug. Isolate it to its own batch.
            logger.warning(
                "picture_enricher.batch_raised",
                extra={"lane": lane, "model": model},
                exc_info=result,
            )
            continue
        written += result
    return written


async def enrich_pictures(document: DoclingDocument, *, max_timeout: float | None = None) -> int:
    """Route document.pictures by classification, write pic.meta.description, return how many.

    Mutates the document in place. Never raises for an LLM or image failure: a picture that
    could not be described keeps meta.description unset, which the chunker drops.

    The stage budget scales with the number of pictures sent, floored by
    PICTURE_ENRICHER_STAGE_TIMEOUT_SECONDS and capped by `max_timeout` (what the task has left).
    On timeout, batches that already finished keep their descriptions.
    """
    _ensure_initialized()
    assert _chart_llm is not None and _caption_llm is not None
    assert _chart_prompt is not None and _caption_prompt is not None
    assert _chart_model is not None and _caption_model is not None

    charts, captions = _plan(document, get_picture_enricher_min_confidence())
    total = len(document.pictures)
    if not charts and not captions:
        logger.info("picture_enricher.nothing_to_do", extra={"pictures": total})
        return 0

    batch_size = get_picture_enricher_batch_size()
    logger.info(
        "picture_enricher.start",
        extra={
            "pictures": total,
            "chart_lane": len(charts),
            "caption_lane": len(captions),
            "skipped": total - len(charts) - len(captions),
            "batch_size": batch_size,
            "chart_model": _chart_model,
            "caption_model": _caption_model,
        },
    )

    # One semaphore across both lanes: the rate limit that matters is the provider account's,
    # not each lane's. Creating it per-lane would let 2x the intended calls in flight.
    sem = asyncio.Semaphore(get_picture_enricher_concurrency())
    attempted = len(charts) + len(captions)
    budget = max(
        get_picture_enricher_stage_timeout(),
        attempted * get_picture_enricher_seconds_per_picture(),
    )
    if max_timeout is not None:
        budget = max(0.0, min(budget, max_timeout))
    lanes = asyncio.gather(
        _run_lane(
            charts,
            llm=_chart_llm,
            model=_chart_model,
            system_prompt=_chart_prompt,
            detail="high",
            batch_size=batch_size,
            lane="chart",
            sem=sem,
        ),
        _run_lane(
            captions,
            llm=_caption_llm,
            model=_caption_model,
            system_prompt=_caption_prompt,
            detail="low",
            batch_size=batch_size,
            lane="caption",
            sem=sem,
        ),
    )
    try:
        chart_written, caption_written = await asyncio.wait_for(lanes, timeout=budget)
        written = chart_written + caption_written
    except TimeoutError:
        models = {_chart_model, _caption_model}
        written = sum(
            1
            for c in (*charts, *captions)
            if c.picture.meta is not None
            and c.picture.meta.description is not None
            and c.picture.meta.description.created_by in models
        )
        logger.warning(
            "picture_enricher.stage_timeout",
            extra={
                "budget_seconds": round(budget, 1),
                "described": written,
                "attempted": attempted,
            },
        )

    logger.info(
        "picture_enricher.done",
        extra={"described": written, "attempted": attempted, "pictures": total},
    )
    return written
