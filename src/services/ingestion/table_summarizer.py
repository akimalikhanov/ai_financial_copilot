"""Generate natural-language summaries for table chunks via the LLM router."""

from __future__ import annotations

import json
import logging
from typing import Any

from src.schemas.table_summarizer import TableSummaryResponse
from src.services.llm_adapters.base_adapter import ChatMessage, Role
from src.services.llm_router import RoutedLLM, get_router
from src.services.prompts.prompt_loader import get_prompt_loader
from src.utils.config import (
    get_table_summarizer_batch_size,
    get_table_summarizer_enable_thinking,
    get_table_summarizer_model,
)
from src.utils.json_schema import build_response_format

logger = logging.getLogger(__name__)

_system_prompt: str | None = None
_llm: RoutedLLM | None = None
_model_id: str | None = None
_enable_thinking: bool = False


# -- Init / reset -------------------------------------------------------------


def _ensure_initialized() -> None:
    """Lazy-init on first use (after fork reset clears stale state)."""
    global _system_prompt, _llm, _model_id, _enable_thinking
    if _llm is not None:
        return
    _system_prompt = get_prompt_loader().load("table_summarizer", "v1").template
    _model_id = get_table_summarizer_model()
    _enable_thinking = get_table_summarizer_enable_thinking()
    _llm = get_router().get(_model_id)


def reset() -> None:
    """Clear cached state (call after fork)."""
    global _system_prompt, _llm, _model_id, _enable_thinking
    _system_prompt = None
    _llm = None
    _model_id = None
    _enable_thinking = False


# -- Batched summarization ----------------------------------------------------


def _build_batch_user_message(batch: list[tuple[int, str]]) -> str:
    """Build a user message containing multiple numbered tables."""
    parts: list[str] = []
    for table_id, enriched_text in batch:
        parts.append(f"[TABLE {table_id}]\n{enriched_text}")
    return "\n\n".join(parts)


def _parse_batch_response(raw: str, batch_ids: list[int]) -> dict[int, str | None]:
    """Parse structured JSON response, returning {table_id: summary} for the batch."""
    result: dict[int, str | None] = dict.fromkeys(batch_ids)
    try:
        data = json.loads(raw)
        parsed = TableSummaryResponse.model_validate(data)
        for item in parsed.summaries:
            if item.table_id in result:
                result[item.table_id] = item.summary
    except Exception:
        logger.warning(
            "table_summarizer.batch_parse_failed", extra={"raw": raw[:500]}, exc_info=True
        )
    return result


async def summarize_table_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add ``table_nl_summary`` and ``table_nl_summary_model`` to every chunk with ``chunk_type == "table"``.

    Tables are batched (BATCH_SIZE per LLM call) with structured output for reliable parsing.
    On per-batch failure, logs a warning and leaves fields as ``None``.
    """
    _ensure_initialized()
    assert _llm is not None and _system_prompt is not None

    table_indices = [i for i, c in enumerate(chunks) if c.get("chunk_type") == "table"]
    if not table_indices:
        return chunks

    batch_size = get_table_summarizer_batch_size()

    logger.info(
        "table_summarizer.start",
        extra={
            "table_count": len(table_indices),
            "batch_size": batch_size,
            "batches": (len(table_indices) + batch_size - 1) // batch_size,
            "model": _model_id,
            "enable_thinking": _enable_thinking,
        },
    )

    extra_params: dict[str, Any] = {"enable_thinking": _enable_thinking}
    response_format = build_response_format(
        "table_summaries", TableSummaryResponse.model_json_schema()
    )

    succeeded = 0
    for batch_start in range(0, len(table_indices), batch_size):
        batch_indices = table_indices[batch_start : batch_start + batch_size]

        # Build numbered batch: use 1-based IDs within each batch
        batch: list[tuple[int, str]] = []
        for local_id, idx in enumerate(batch_indices, start=1):
            batch.append((local_id, chunks[idx]["enriched_text"]))

        batch_ids = [tid for tid, _ in batch]
        user_msg = _build_batch_user_message(batch)
        messages = [
            ChatMessage(role=Role.system, content=_system_prompt),
            ChatMessage(role=Role.user, content=user_msg),
        ]

        try:
            resp = await _llm.complete(
                messages,
                temperature=0.0,
                max_tokens=300 * len(batch_indices),
                response_format=response_format,
                **extra_params,
            )
            summaries = _parse_batch_response(resp.text, batch_ids)
        except Exception:
            logger.warning(
                "table_summarizer.batch_failed",
                extra={
                    "batch_start": batch_start,
                    "batch_size": len(batch_indices),
                    "model": _model_id,
                },
                exc_info=True,
            )
            summaries = dict.fromkeys(batch_ids)

        # Map summaries back to chunks
        for local_id, idx in enumerate(batch_indices, start=1):
            summary = summaries.get(local_id)
            chunks[idx]["table_nl_summary"] = summary
            chunks[idx]["table_nl_summary_model"] = _model_id if summary else None
            if summary:
                succeeded += 1

    logger.info(
        "table_summarizer.done",
        extra={
            "succeeded": succeeded,
            "failed": len(table_indices) - succeeded,
            "model": _model_id,
        },
    )
    return chunks
