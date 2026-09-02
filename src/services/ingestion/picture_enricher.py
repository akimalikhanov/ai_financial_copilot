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

import io
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.schemas.picture_enricher import PictureDescriptionResponse
from src.services.llm_adapters.base_adapter import ChatMessage, ImagePart, Role
from src.services.llm_router import RoutedLLM, get_router
from src.services.prompts.prompt_loader import get_prompt_loader
from src.utils.config import (
    get_picture_enricher_batch_size,
    get_picture_enricher_cheap_model,
    get_picture_enricher_min_confidence,
    get_picture_enricher_model,
)
from src.utils.json_schema import build_response_format

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument, PictureItem

logger = logging.getLogger(__name__)

# Lanes, keyed by the 16 labels docling's document figure classifier actually emits. Anything
# unlisted (including a label from a future model revision) falls through to CAPTION, so a new
# class degrades to a cheap caption rather than to silence.
CHART_LABELS = frozenset({"bar_chart", "line_chart", "pie_chart", "flow_chart"})
# No model call: these carry no retrievable content, and a description of a signature or a
# decorative icon is noise that gets embedded and BM25-indexed as document text.
SKIP_LABELS = frozenset({"logo", "icon", "signature", "stamp", "qr_code", "bar_code"})

_MIME = "image/png"  # what export writes; see tasks._extract_picture_crops
# One retry: the chart lane occasionally returns HTTP 200 with an empty body.
_MAX_ATTEMPTS = 2


@dataclass(frozen=True, slots=True)
class _Candidate:
    """One picture to describe, paired with the crop bytes and the lane it routed to."""

    picture: PictureItem
    label: str
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

    Same source as tasks._extract_picture_crops: the crops exist on the document only because
    generate_picture_images=True, and only until export strips them.
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
    skip labels and anything with no crop to send."""
    charts: list[_Candidate] = []
    captions: list[_Candidate] = []
    for pic in document.pictures:
        label = _label_of(pic, min_confidence)
        if label in SKIP_LABELS:
            continue
        try:
            image = _crop_bytes(pic)
        except Exception:
            logger.warning(
                "picture_enricher.crop_failed",
                extra={"self_ref": pic.self_ref},
                exc_info=True,
            )
            continue
        if not image:
            continue
        candidate = _Candidate(picture=pic, label=label, image=image)
        (charts if label in CHART_LABELS else captions).append(candidate)
    return charts, captions


# -- LLM calls ----------------------------------------------------------------


def _build_message(batch: list[_Candidate], detail: str) -> ChatMessage:
    """One user message holding the whole batch: a text part naming each picture, then the
    crops in the same order. The model sees only pixels and this text, so the label has to be
    written in explicitly (see docs/notes/llm-image-inputs.md)."""
    lines = [f"[PICTURE {i}] {c.label}" for i, c in enumerate(batch, start=1)]
    return ChatMessage(
        role=Role.user,
        content="\n".join(lines),
        images=tuple(ImagePart(data=c.image, mime_type=_MIME, detail=detail) for c in batch),  # type: ignore[arg-type]
    )


def _parse(raw: str, size: int) -> dict[int, str] | None:
    """Parse structured output into {1-based picture id: description}, dropping blanks.

    Returns None when the response could not be parsed at all — an empty body or malformed
    JSON, both of which are model failures worth one retry. A parsed response with empty
    descriptions returns an empty mapping instead: that is the prompt's escape hatch for an
    unreadable image, and retrying it would just pay twice for the same answer.
    """
    if not raw.strip():
        logger.warning("picture_enricher.empty_response")
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
    batch: list[_Candidate],
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
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            resp = await llm.complete(
                messages,
                _lf_name=f"picture_enricher.{lane}",
                temperature=0.0,
                # On GPT-5 models this maps to max_completion_tokens, which counts reasoning
                # tokens too — the description gets what is left after reasoning.
                max_tokens=1200 * len(batch),
                response_format=response_format,
            )
        except Exception:
            logger.warning(
                "picture_enricher.batch_call_failed",
                extra={"lane": lane, "model": model, "attempt": attempt},
                exc_info=True,
            )
        else:
            descriptions = _parse(resp.text, len(batch))
            if descriptions is not None:
                return descriptions
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
) -> int:
    """Describe one lane's pictures in batches, writing meta.description in place."""
    if not candidates:
        return 0

    from docling_core.types.doc.document import DescriptionMetaField, PictureMeta

    response_format = build_response_format(
        "picture_descriptions", PictureDescriptionResponse.model_json_schema()
    )
    written = 0
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        descriptions = await _describe_batch(
            batch,
            llm=llm,
            model=model,
            system_prompt=system_prompt,
            detail=detail,
            lane=lane,
            response_format=response_format,
        )
        if descriptions is None:
            # Per-batch isolation: the contract is that enrichment degrades, never fails.
            logger.warning(
                "picture_enricher.batch_failed",
                extra={
                    "lane": lane,
                    "batch_start": start,
                    "batch_size": len(batch),
                    "model": model,
                },
            )
            continue

        for local_id, candidate in enumerate(batch, start=1):
            text = descriptions.get(local_id)
            if not text:
                continue
            pic = candidate.picture
            if pic.meta is None:
                pic.meta = PictureMeta()
            pic.meta.description = DescriptionMetaField(text=text, created_by=model)
            written += 1
    return written


async def enrich_pictures(document: DoclingDocument) -> int:
    """Route document.pictures by classification, write pic.meta.description, return how many.

    Mutates the document in place. Never raises for an LLM or image failure: a picture that
    could not be described keeps meta.description unset, which the chunker drops.
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

    written = await _run_lane(
        charts,
        llm=_chart_llm,
        model=_chart_model,
        system_prompt=_chart_prompt,
        detail="high",
        batch_size=batch_size,
        lane="chart",
    )
    written += await _run_lane(
        captions,
        llm=_caption_llm,
        model=_caption_model,
        system_prompt=_caption_prompt,
        detail="low",
        batch_size=batch_size,
        lane="caption",
    )

    logger.info(
        "picture_enricher.done",
        extra={"described": written, "attempted": len(charts) + len(captions), "pictures": total},
    )
    return written
