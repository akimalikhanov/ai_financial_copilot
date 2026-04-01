from __future__ import annotations

import logging
import re
from datetime import date
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

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

logger = logging.getLogger(__name__)

_FALLBACK = RouterOutput(
    route="retrieval",
    route_confidence=0.5,
    entities=[],
    time_references=[],
    doc_type_hints=[],
    user_intent="fallback",
    needs_decomposition=False,
    reasoning="router_fallback",
)


def _normalize(raw: str) -> str:
    cleaned = re.sub(r"\s+", " ", raw.strip())
    if not cleaned:
        raise ValueError("Query cannot be empty")
    return cleaned


def _make_strict(obj: dict) -> None:
    """Mutate a JSON schema object in-place for OpenAI strict mode.

    Strict mode requires:
    - additionalProperties: false on every object
    - required lists every key in properties (including optional/defaulted ones)
    """
    if obj.get("type") == "object" and "properties" in obj:
        obj["additionalProperties"] = False
        obj["required"] = list(obj["properties"].keys())
        for prop in obj["properties"].values():
            if isinstance(prop, dict):
                _make_strict(prop)
    for key in ("anyOf", "oneOf", "allOf"):
        for sub in obj.get(key, []):
            if isinstance(sub, dict):
                _make_strict(sub)
    for def_schema in obj.get("$defs", {}).values():
        if isinstance(def_schema, dict):
            _make_strict(def_schema)


def _router_response_format() -> dict:
    schema = RouterOutput.model_json_schema()
    _make_strict(schema)

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "query_router",
            "schema": schema,
            "strict": True,
        },
    }


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
    if inp.scope is not None and inp.scope.mode == "filteredByMetadata":
        f = inp.scope.filters
        parts = []
        if f.company:
            parts.append(f"company={', '.join(f.company)}")
        if f.year:
            parts.append(f"year={', '.join(str(y) for y in f.year)}")
        if parts:
            scope_block = f"Active document scope: {'; '.join(parts)}\n"

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
) -> tuple[RouterOutput, DocumentScopeResult | None]:
    """Route a query and optionally resolve document scope.

    Returns (RouterOutput, DocumentScopeResult | None).
    scope_result is None when route != 'retrieval' or session is not provided.
    """
    normalized = _normalize(inp.query)
    inp = inp.model_copy(update={"query": normalized})

    router = llm_router if llm_router is not None else get_router()
    model_id = get_query_router_model()

    try:
        llm = router.get(model_id)
    except Exception:
        logger.warning("route_query_model_unavailable", extra={"model": model_id})
        return _FALLBACK, None

    try:
        prompt = get_prompt_loader().load("query_router", "v2")
        system = get_prompt_renderer()._render_template(
            prompt.template, {"today": date.today().isoformat()}
        )
    except Exception:
        logger.warning("route_query_prompt_missing", extra={"model": model_id})
        return _FALLBACK, None

    response_format = _router_response_format()
    messages = _build_messages(inp, system, max_assistant_tokens=150)

    cfg = get_router_config()
    output: RouterOutput | None = None
    for attempt in range(2):
        try:
            resp = await llm.complete(
                messages=messages,
                temperature=cfg["temperature"],
                max_tokens=int(cfg["max_tokens"]),
                response_format=response_format,
            )
        except Exception as e:
            logger.exception("route_query_llm_error", extra={"error": str(e)})
            return _FALLBACK, None

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

    # Scope-based overrides (post-LLM, deterministic)
    _has_active_scope = inp.scope is not None and (
        inp.scope.mode in ("selectedDocs", "thisDoc")
        or (
            inp.scope.mode == "filteredByMetadata"
            and (inp.scope.filters.company or inp.scope.filters.year)
        )
    )

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
        "route_query_done route=%s confidence=%.2f entities=%d",
        output.route,
        output.route_confidence,
        len(output.entities),
    )

    if session is not None and user_id is not None and output.route == "retrieval":
        scope_result = await resolve_scope(session, user_id, inp.scope, output)
        return output, scope_result

    return output, None
