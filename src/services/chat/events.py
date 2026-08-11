"""Event helpers for the chat pipeline. Pure functions, no side effects."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Literal
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from src.repository.chunk_repository import ChunkRepository
from src.schemas.retrieval import AnswerCitationSpan, ChunkProvenance, Citation, RAGContext
from src.services.llm_adapters.base_adapter import LLMResponseStats
from src.services.retrieval.payload_hydrator import _parse_provenance

logger = logging.getLogger(__name__)


class ThinkingStripper:
    """Strips <think>...</think> blocks from streaming text (e.g. Qwen 3 reasoning tokens)."""

    def __init__(self) -> None:
        self._in_think = False
        self._strip_leading_newlines = False
        self._buf = ""

    def feed(self, text: str) -> str:
        """Process a chunk; returns the text with thinking blocks removed."""
        self._buf += text
        out: list[str] = []
        while self._buf:
            if self._in_think:
                end = self._buf.find("</think>")
                if end == -1:
                    self._buf = ""
                    break
                self._in_think = False
                self._strip_leading_newlines = True
                self._buf = self._buf[end + 8 :]
            else:
                if self._strip_leading_newlines:
                    self._buf = self._buf.lstrip("\n")
                    self._strip_leading_newlines = False
                    if not self._buf:
                        break
                start = self._buf.find("<think>")
                if start == -1:
                    out.append(self._buf)
                    self._buf = ""
                    break
                out.append(self._buf[:start])
                self._in_think = True
                self._buf = self._buf[start + 7 :]
        return "".join(out)


def out_of_scope_response() -> str:
    """Fixed redirect message for out-of-scope queries."""
    return (
        "I'm focused on financial document analysis and can't help with that. "
        "Feel free to ask about financial reports, filings, or documents you've uploaded."
    )


_GENERIC_USER_ERROR = "Something went wrong. Please try again."


def error_event(exc: Exception, user_message: str | None = None) -> dict:
    """Structured error event for frontend display."""
    from src.services.llm_runtime.exceptions import LLMError

    if user_message is None and isinstance(exc, LLMError):
        payload = exc.to_dict(as_json=False)
        if isinstance(payload, dict):
            user_message = payload.get("user_message")

    return {
        "error_type": type(exc).__name__,
        "message": str(exc),
        "user_message": user_message or _GENERIC_USER_ERROR,
    }


ActivityKind = Literal[
    "stage_started",
    "stage_ended",
    "tool_call_started",
    "tool_call_ended",
    "round_started",
]


def build_activity_event(
    kind: ActivityKind,
    *,
    event_id: str | None = None,
    label: str | None = None,
    parent_id: str | None = None,
    detail: dict | None = None,
) -> tuple[str, dict]:
    """Build an `activity` event payload. Returns (id, event_data) — the id is the
    caller's handle for emitting the matching `_ended` event later (e.g. `stage_ended`
    or `tool_call_ended` reference the `_started` event's id, not its label), so
    correlation doesn't depend on label/entity string matching under concurrency.

    Pass `event_id` when closing a previously-started activity — it must be the id
    returned for that activity's `_started` event, not a fresh one."""
    event_id = event_id or str(uuid.uuid4())
    data: dict = {"kind": kind, "id": event_id, "ts": time.time()}
    if parent_id is not None:
        data["parent_id"] = parent_id
    if label is not None:
        data["label"] = label
    if detail is not None:
        data["detail"] = detail
    return event_id, data


def _provenance_bbox_hints(provenance: ChunkProvenance | None) -> list[dict] | None:
    """Compute a union bounding box per page across all provenance items on that page.

    A chunk may span multiple text blocks per page (and multiple pages); merging
    each page's blocks into one rect per page gives a highlight that covers the
    whole chunk on every page it touches, without multiple overlapping overlays.
    Docling bbox coordinates are in absolute PDF points (typically BOTTOMLEFT origin).
    """
    if not provenance:
        return None
    items_with_bbox = [item for item in provenance.items if item.bbox is not None]
    if not items_with_bbox:
        return None
    pages = sorted({item.page_no for item in items_with_bbox})
    hints: list[dict] = []
    for page_no in pages:
        page_items = [item for item in items_with_bbox if item.page_no == page_no]
        coord_origin = page_items[0].bbox.coord_origin  # type: ignore[union-attr]
        left = min(item.bbox.left for item in page_items)  # type: ignore[union-attr]
        right = max(item.bbox.right for item in page_items)  # type: ignore[union-attr]
        # For BOTTOMLEFT: bottom < top numerically; union keeps the lower bottom and higher top
        bottom = min(item.bbox.bottom for item in page_items)  # type: ignore[union-attr]
        top = max(item.bbox.top for item in page_items)  # type: ignore[union-attr]
        hints.append(
            {
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
                "coord_origin": coord_origin,
                "page": page_no,
            }
        )
    return hints


def citation_to_dict(c: Citation, provenance: ChunkProvenance | None = None) -> dict:
    """Serialize Citation for JSON event payload."""
    d: dict = {
        "ref_id": c.ref_id,
        "ref_index": c.ref_index,
        "chunk_id": str(c.chunk_id),
        "document_id": str(c.document_id),
        "document_name": c.document_name,
        "filename": c.filename,
        "page_numbers": list(c.page_numbers),
        "heading_path": list(c.heading_path),
        "snippet": c.snippet,
    }
    bbox_hints = _provenance_bbox_hints(provenance)
    if bbox_hints is not None:
        d["bbox_hints"] = bbox_hints
    return d


def build_references_list(rag_context: RAGContext, cited_ref_ids: list[str]) -> list[dict]:
    """Build the references list for the sources the answer actually cited, ordered by
    first appearance in the answer."""
    item_by_id = {item.ref_id: item for item in rag_context.items}
    result: list[dict] = []
    unresolved: list[str] = []
    for ref_id in cited_ref_ids:
        ctx_item = item_by_id.get(ref_id)
        if ctx_item:
            entry = citation_to_dict(ctx_item.citation, ctx_item.provenance)
            entry["display_label"] = ref_id
            result.append(entry)
        else:
            unresolved.append(ref_id)
    if unresolved:
        # The model cited a source ID with no matching context item — the citation
        # pill for it will render but won't resolve to an evidence entry.
        logger.warning(
            "cited_refs_missing_from_context",
            extra={
                "unresolved_refs": unresolved,
                "context_ref_ids": sorted(item_by_id),
            },
        )
    return result


def span_to_dict(span: AnswerCitationSpan) -> dict:
    """Serialize an AnswerCitationSpan for JSON event payload."""
    return {"start": span.start, "end": span.end, "ref_ids": list(span.ref_ids)}


def build_all_references(rag_context: RAGContext) -> list[dict]:
    """Build references from ALL RAG items — the fallback when the model emitted no
    parseable citations at all.

    Sorting by score agrees with the S-labels here: the agent path now assigns labels in
    global score order, so S1 really is the highest-scoring chunk. The display label keeps
    the original S-prefix so it matches any S-labels the model wrote naturally.
    """
    sorted_items = sorted(rag_context.items, key=lambda i: i.score, reverse=True)
    result = []
    for item in sorted_items:
        entry = citation_to_dict(item.citation, item.provenance)
        entry["display_label"] = item.citation.ref_id  # S1, S2, ... by relevance order
        result.append(entry)
    return result


def build_usage_event(
    assistant_message_id: UUID,
    assistant_seq: int,
    stats: LLMResponseStats | None,
    citation_spans: list[AnswerCitationSpan] | None = None,
    references: list[dict] | None = None,
) -> dict:
    """Build usage_data dict for the usage event."""
    usage_data: dict = {
        "persisted": True,
        "assistant_message_id": str(assistant_message_id),
        "assistant_seq": assistant_seq,
    }
    if stats:
        usage_data["stats"] = {
            "input_tokens": stats.input_tokens,
            "output_tokens": stats.output_tokens,
            "reasoning_tokens": stats.reasoning_tokens,
            "total_tokens": stats.total_tokens,
            "latency_ms": stats.latency_ms,
            "ttft_ms": stats.ttft_ms,
            "tps": stats.tps,
            "cost_usd": stats.cost_usd,
        }

    if citation_spans:
        usage_data["citation_spans"] = [span_to_dict(sp) for sp in citation_spans]
    if references:
        usage_data["references"] = references

    return usage_data


async def hydrate_bbox_hints(
    session: AsyncSession,
    messages_metadata: list[dict],
) -> None:
    """Backfill ``bbox_hints`` on persisted message references for old messages.

    Mutates each metadata dict in place. Looks at ``references`` and ``citations``
    arrays, collects chunk_ids missing ``bbox_hints``, batch-fetches chunks, and
    injects bbox_hints computed from chunk provenance.
    """
    needed: set[UUID] = set()
    targets: list[dict] = []
    for meta in messages_metadata:
        if not isinstance(meta, dict):
            continue
        for key in ("references", "citations"):
            arr = meta.get(key)
            if not isinstance(arr, list):
                continue
            for entry in arr:
                if not isinstance(entry, dict) or "bbox_hints" in entry:
                    continue
                cid = entry.get("chunk_id")
                if not cid:
                    continue
                try:
                    needed.add(UUID(cid))
                    targets.append(entry)
                except (ValueError, AttributeError, TypeError):
                    continue

    if not needed:
        return

    chunk_repo = ChunkRepository(session)
    chunks = await chunk_repo.get_by_ids(list(needed))
    bbox_by_chunk: dict[str, list[dict]] = {}
    for chunk in chunks:
        prov = _parse_provenance(chunk.provenance if isinstance(chunk.provenance, list) else None)
        hints = _provenance_bbox_hints(prov)
        if hints is not None:
            bbox_by_chunk[str(chunk.id)] = hints

    for entry in targets:
        cid = entry.get("chunk_id")
        if isinstance(cid, str) and cid in bbox_by_chunk:
            entry["bbox_hints"] = bbox_by_chunk[cid]
