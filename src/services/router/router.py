from __future__ import annotations

import logging
import re
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from src.repository.llm_request_repository import LLMRequestRepository, stats_to_request_kwargs
from src.schemas.query_router import (
    DocumentScopeResult,
    RouterInput,
    RouterOutput,
)
from src.services.llm_adapters.base_adapter import ChatMessage, Role
from src.services.llm_router import LLMRouter, get_router
from src.services.prompts.prompt_loader import get_prompt_loader
from src.services.prompts.prompt_renderer import get_prompt_renderer
from src.services.router.parser import parse_router_response
from src.services.router.scope_resolver import resolve_scope
from src.utils.config import get_query_router_model, get_router_config
from src.utils.json_schema import build_response_format

logger = logging.getLogger(__name__)

_FALLBACK = RouterOutput(
    route="retrieval",
    entities=[],
    user_intent="fallback",
    needs_decomposition=False,
    reasoning="router_fallback",
)


def _normalize(raw: str) -> str:
    cleaned = re.sub(r"\s+", " ", raw.strip())
    if not cleaned:
        raise ValueError("Query cannot be empty")
    return cleaned


def _router_response_format() -> dict:
    return build_response_format("query_router", RouterOutput.model_json_schema())


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Rough token estimation: ~1 token per 4 chars. Truncates aggressively to stay under limit."""
    token_count = len(text) // 4
    if token_count <= max_tokens:
        return text
    char_limit = max_tokens * 4
    return text[:char_limit].rstrip() + "..."


def _build_messages(
    inp: RouterInput, system: str, max_assistant_tokens: int = 150
) -> list[ChatMessage]:
    """Build router messages with assistant turns truncated to token budget.

    Args:
        inp: Router input with query and conversation history
        system: System prompt
        max_assistant_tokens: Max tokens per assistant turn (default 150 per spec)
    """
    history_block = ""
    if inp.conversation_history:
        turns = []
        for turn in inp.conversation_history:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            # Truncate assistant turns to stay within token budget
            if role == "assistant":
                content = _truncate_to_tokens(content, max_assistant_tokens)
            turns.append(f"{role}: {content}")
        history_block = "Recent conversation:\n" + "\n".join(turns) + "\n\n"

    scope_block = ""
    if inp.scope is not None:
        if inp.scope.mode == "filteredByMetadata":
            f = inp.scope.filters
            parts = []
            if f.company:
                parts.append(f"company={', '.join(f.company)}")
            if f.year:
                parts.append(f"year={', '.join(str(y) for y in f.year)}")
            if parts:
                scope_block = f"Active document scope: {'; '.join(parts)}\n"
        elif inp.scope.mode in ("selectedDocs", "thisDoc"):
            scope_block = "Active document scope: specific documents explicitly selected by user\n"

    return [
        ChatMessage(role=Role.system, content=system),
        ChatMessage(role=Role.user, content=f"{scope_block}{history_block}User query: {inp.query}"),
    ]


async def route_query(
    inp: RouterInput,
    *,
    user_id: UUID | None = None,
    llm_router: LLMRouter | None = None,
    session: AsyncSession | None = None,
    parent_request_id: UUID | None = None,
    conversation_id: UUID | None = None,
) -> tuple[RouterOutput, DocumentScopeResult | None]:
    """Route a query and optionally resolve document scope.

    Returns (RouterOutput, DocumentScopeResult | None).
    scope_result is None when route != 'retrieval' or session is not provided.

    When `parent_request_id`, `conversation_id`, and `session` are provided, the
    router's LLM call(s) are logged to llm_requests as a sub-request of the parent
    chat request, so full-pipeline cost/tokens can be aggregated.
    """
    normalized = _normalize(inp.query)
    inp = inp.model_copy(update={"query": normalized})

    router = llm_router if llm_router is not None else get_router()
    model_id = get_query_router_model()
    should_log_subrequest = (
        session is not None and parent_request_id is not None and conversation_id is not None
    )

    try:
        llm = router.get(model_id)
    except Exception:
        logger.warning("route_query_model_unavailable", extra={"model": model_id})
        return _FALLBACK, None

    try:
        prompt = get_prompt_loader().load("query_router", "v2")
        system = get_prompt_renderer()._render_template(prompt.template, {})
    except Exception:
        logger.warning("route_query_prompt_missing", extra={"model": model_id})
        return _FALLBACK, None

    response_format = _router_response_format()
    messages = _build_messages(inp, system, max_assistant_tokens=150)

    cfg = get_router_config()
    request_params = {
        "temperature": cfg["temperature"],
        "max_tokens": int(cfg["max_tokens"]),
    }
    output: RouterOutput | None = None
    for attempt in range(2):
        try:
            resp = await llm.complete(
                messages=messages,
                _lf_name="query_router",
                temperature=cfg["temperature"],
                max_tokens=int(cfg["max_tokens"]),
                response_format=response_format,
            )
        except Exception as e:
            logger.exception("route_query_llm_error", extra={"error": str(e)})
            return _FALLBACK, None

        if should_log_subrequest:
            await LLMRequestRepository(session).create_subrequest(  # type: ignore[arg-type]
                parent_request_id=parent_request_id,  # type: ignore[arg-type]
                conversation_id=conversation_id,  # type: ignore[arg-type]
                user_id=user_id,
                provider=llm.provider,
                model=model_id,
                request_type="router",
                request_params=request_params,
                status="completed",
                **stats_to_request_kwargs(resp.stats),
            )

        raw = resp.text or ""
        result, error = parse_router_response(raw)
        if result is not None:
            output = result
            break

        logger.warning(
            "route_query_parse_failed attempt=%d error=%s raw=%r",
            attempt,
            error,
            raw[:300],
        )
        if attempt == 0:
            retry_content = (
                f"User query: {inp.query}\n\n"
                f"Previous response: {raw[:300]}\n"
                f"Error: {error}\n\n"
                "Fix the error. Return ONLY valid JSON matching the required schema."
            )
            messages = [
                ChatMessage(role=Role.system, content=system),
                ChatMessage(role=Role.user, content=retry_content),
            ]

    if output is None:
        return _FALLBACK, None

    # if output.route == "retrieval" and not output.entities and not _has_active_scope:
    #     # Retrieval with no entities and no scope — ask user to be more specific
    #     logger.info("route_query_override route=retrieval→direct_answer reason=no_entity_no_scope")
    #     output = output.model_copy(update={"route": "direct_answer"})
    # elif output.route != "retrieval" and _has_active_scope:
    #     # Active scope means the user is explicitly targeting documents — always retrieve
    #     logger.info(
    #         "route_query_override route=%s→retrieval reason=active_scope",
    #         output.route,
    #     )
    #     output = output.model_copy(update={"route": "retrieval", "route_confidence": 1.0})

    logger.info(
        "route_query_done route=%s entities=%d",
        output.route,
        len(output.entities),
    )

    if session is not None and user_id is not None and output.route == "retrieval":
        scope_result = await resolve_scope(session, user_id, inp.scope, output)
        return output, scope_result

    return output, None
