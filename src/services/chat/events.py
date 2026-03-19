"""Event helpers for the chat pipeline. Pure functions, no side effects."""

from __future__ import annotations

import re
from uuid import UUID

from src.schemas.retrieval import Citation, RAGContext
from src.services.llm_adapters.base_adapter import LLMResponseStats


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
    """Parse [C1], [C2], etc. from LLM output and map to citation objects.
    Only matches ref_ids that exist in rag_context (whitelist) to avoid false positives."""
    ref_by_id = {item.ref_id: item.citation for item in rag_context.items}
    if not ref_by_id:
        return []
    valid_ref_ids = sorted(ref_by_id, key=lambda r: (-len(r), r))  # longer first (C10 before C1)
    pattern = re.compile(r"\[(" + "|".join(re.escape(r) for r in valid_ref_ids) + r")\]")
    seen: set[str] = set()
    result: list[Citation] = []
    for m in pattern.finditer(text):
        ref_id = m.group(1)
        if ref_id not in seen:
            seen.add(ref_id)
            result.append(ref_by_id[ref_id])
    return result


def build_usage_event(
    accumulated_content: str,
    rag_context: RAGContext | None,
    assistant_message_id: UUID,
    assistant_seq: int,
    stats: LLMResponseStats | None,
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
    if rag_context and rag_context.items:
        used = extract_used_citations(accumulated_content, rag_context)
        usage_data["citations"] = [citation_to_dict(c) for c in used]
    return usage_data
