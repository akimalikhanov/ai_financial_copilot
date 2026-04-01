"""Event helpers for the chat pipeline. Pure functions, no side effects."""

from __future__ import annotations

import re
from uuid import UUID

from src.schemas.retrieval import AnswerCitationSpan, Citation, DisplayLabelMap, RAGContext
from src.services.llm_adapters.base_adapter import LLMResponseStats


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


def error_event(exc: Exception, user_message: str | None = None) -> dict:
    """Structured error event for frontend display."""
    return {
        "error_type": type(exc).__name__,
        "message": str(exc),
        "user_message": user_message or str(exc),
    }


def citation_to_dict(c: Citation) -> dict:
    """Serialize Citation for JSON event payload."""
    return {
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


def extract_used_citations(text: str, rag_context: RAGContext) -> list[Citation]:
    """Parse [S1], [S2], [C1], [C2], etc. from LLM output and map to citation objects.

    Matches both S-prefixed (new) and C-prefixed (legacy) ref_ids that exist
    in rag_context (whitelist) to avoid false positives.
    """
    ref_by_id = {item.ref_id: item.citation for item in rag_context.items}
    if not ref_by_id:
        return []
    valid_ref_ids = sorted(ref_by_id, key=lambda r: (-len(r), r))  # longer first (S10 before S1)
    pattern = re.compile(r"\[(" + "|".join(re.escape(r) for r in valid_ref_ids) + r")\]")
    seen: set[str] = set()
    result: list[Citation] = []
    for m in pattern.finditer(text):
        ref_id = m.group(1)
        if ref_id not in seen:
            seen.add(ref_id)
            result.append(ref_by_id[ref_id])
    return result


def build_references_list(
    rag_context: RAGContext,
    label_map: DisplayLabelMap,
) -> list[dict]:
    """Build ordered references list from label map and RAG context.

    Returns references sorted by display label (C1, C2, ...).
    Only includes sources that were actually cited in the answer.
    """
    ref_by_id = {item.ref_id: item.citation for item in rag_context.items}
    result: list[dict] = []
    for source_id, display_label in sorted(
        label_map.mapping.items(),
        key=lambda x: int(x[1][1:]),  # sort by numeric part of "C1", "C2", ...
    ):
        citation = ref_by_id.get(source_id)
        if citation:
            entry = citation_to_dict(citation)
            entry["display_label"] = display_label
            result.append(entry)
    return result


def span_to_dict(span: AnswerCitationSpan, display_labels: tuple[str, ...]) -> dict:
    """Serialize an AnswerCitationSpan for JSON event payload."""
    return {
        "start": span.start,
        "end": span.end,
        "ref_ids": list(span.ref_ids),
        "display_labels": list(display_labels),
    }


def build_all_references(rag_context: RAGContext) -> list[dict]:
    """Build references from ALL RAG items for weak models (citation_mode: none).

    Items are ordered by reranker score (S1 = highest). The display label keeps
    the original S-prefix (S1, S2, ...) so it matches any S-labels the model
    may have written naturally in its answer, since weak models see those labels
    in the context but are not instructed to format them as bracket citations.
    """
    sorted_items = sorted(rag_context.items, key=lambda i: i.score, reverse=True)
    result = []
    for item in sorted_items:
        entry = citation_to_dict(item.citation)
        entry["display_label"] = item.citation.ref_id  # S1, S2, ... by relevance order
        result.append(entry)
    return result


def build_usage_event(
    accumulated_content: str,
    rag_context: RAGContext | None,
    assistant_message_id: UUID,
    assistant_seq: int,
    stats: LLMResponseStats | None,
    citation_spans: list[AnswerCitationSpan] | None = None,
    label_map: DisplayLabelMap | None = None,
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

    if citation_spans and label_map:
        # New structured citation spans
        usage_data["citation_spans"] = [
            span_to_dict(s, label_map.get_labels_for_refs(s.ref_ids)) for s in citation_spans
        ]

    if rag_context and rag_context.items:
        if label_map and label_map.mapping:
            # Build references from label map (only cited sources)
            usage_data["references"] = build_references_list(rag_context, label_map)
            # Backwards-compat: also include flat citations list
            usage_data["citations"] = usage_data["references"]
        else:
            # Regex fallback: no <claim> tags were found, extract [S1] or [C1] patterns
            used = extract_used_citations(accumulated_content, rag_context)
            usage_data["citations"] = [citation_to_dict(c) for c in used]

    return usage_data
