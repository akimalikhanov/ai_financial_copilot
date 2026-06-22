"""Auto-generate a short conversation title from the first user query."""

from __future__ import annotations

import logging
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from src.repository.llm_request_repository import LLMRequestRepository, stats_to_request_kwargs
from src.services.llm_adapters.base_adapter import ChatMessage, Role
from src.services.llm_router import LLMRouter

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You generate ultra-short titles for chat conversations. "
    "Reply with ONLY the title — no quotes, no punctuation at the end, no explanation. "
    "Maximum {max_len} characters."
)
_USER = "First message: {query}\n\nGenerate a title."


async def generate_conversation_title(
    query: str,
    *,
    llm_router: LLMRouter,
    model: str,
    max_len: int = 60,
    session: AsyncSession | None = None,
    parent_request_id: UUID | None = None,
    conversation_id: UUID | None = None,
    user_id: UUID | None = None,
) -> str | None:
    """Call a cheap LLM to produce a short title. Returns None on any failure.

    Registers a Langfuse generation (via ``_lf_name``) and, when the logging
    context is supplied, a completed ``llm_requests`` sub-request — matching the
    pattern used by rewrite_query / query_router / table_summarizer.
    """
    request_params: dict = {"temperature": 0.3, "max_tokens": 30}
    should_log_subrequest = (
        session is not None and parent_request_id is not None and conversation_id is not None
    )

    try:
        llm = llm_router.get(model)
    except Exception:
        logger.warning("conversation_naming_model_unavailable", extra={"model": model})
        return None

    messages = [
        ChatMessage(role=Role.system, content=_SYSTEM.format(max_len=max_len)),
        ChatMessage(role=Role.user, content=_USER.format(query=query[:500])),
    ]

    try:
        resp = await llm.complete(
            messages=messages,
            _lf_name="conversation_naming",
            temperature=request_params["temperature"],
            max_tokens=request_params["max_tokens"],
        )
    except Exception:
        logger.warning("conversation_naming_llm_error", exc_info=True)
        return None

    if should_log_subrequest:
        await LLMRequestRepository(session).create_subrequest(  # type: ignore[arg-type]
            parent_request_id=parent_request_id,  # type: ignore[arg-type]
            conversation_id=conversation_id,  # type: ignore[arg-type]
            user_id=user_id,
            provider=llm.provider,
            model=model,
            request_type="conversation_naming",
            request_params=request_params,
            status="completed",
            **stats_to_request_kwargs(resp.stats),
        )

    title = (resp.text or "").strip().strip("\"'").strip()[:max_len]
    return title if title else None
