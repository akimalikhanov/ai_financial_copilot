from __future__ import annotations

import json
import logging
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from src.repository.llm_request_repository import LLMRequestRepository, stats_to_request_kwargs
from src.schemas.query_transform import SubQuery, TransformedQuery, TransformerInput
from src.services.llm_adapters.base_adapter import ChatMessage, Role
from src.services.llm_router import LLMRouter, get_router
from src.services.prompts.prompt_loader import get_prompt_loader
from src.services.prompts.prompt_renderer import get_prompt_renderer
from src.utils.config import get_query_transformer_config, get_query_transformer_model
from src.utils.json_schema import build_response_format

logger = logging.getLogger(__name__)

TRANSFORMER_CONV_HISTORY_TOKENS = 1200


def _fallback(raw_query: str) -> TransformedQuery:
    return TransformedQuery(
        semantic_query=raw_query,
        keyword_query=raw_query,
        fallback=True,
    )


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """~1 token per 4 chars. Truncates aggressively to stay under limit."""
    if len(text) // 4 <= max_tokens:
        return text
    return text[: max_tokens * 4].rstrip() + "..."


def _transformer_response_format() -> dict:
    # Build schema with only LLM-facing fields — runtime-only fields (fallback,
    # decomposition_overridden) are stripped by _parse_transformer_response and must
    # not appear in required[], which was causing OpenAI to truncate after sub_queries=[].
    schema = TransformedQuery.model_json_schema()
    runtime_fields = {"fallback", "decomposition_overridden"}
    for field in runtime_fields:
        schema.get("properties", {}).pop(field, None)
    if "required" in schema:
        schema["required"] = [f for f in schema["required"] if f not in runtime_fields]
    return build_response_format("query_transformer", schema)


def _build_messages(inp: TransformerInput, system: str) -> list[ChatMessage]:
    parts: list[str] = []

    if inp.scope_docs:
        lines = ["Active documents:"]
        for doc in inp.scope_docs:
            meta = f"  [{doc.document_id}]"
            if doc.company:
                meta += f" company={doc.company}"
            if doc.year:
                meta += f" | year={doc.year}"
            lines.append(meta)
        parts.append("\n".join(lines))

    if inp.known_entity_names:
        lines = ["Known entities (use these EXACTLY as focus_entity values):"]
        for i, name in enumerate(inp.known_entity_names):
            lines.append(f"  [{i}] {name}")
        parts.append("\n".join(lines))

    if inp.conversation_history:
        turns = []
        for turn in inp.conversation_history:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role == "assistant":
                content = _truncate_to_tokens(content, TRANSFORMER_CONV_HISTORY_TOKENS)
            turns.append(f"{role}: {content}")
        parts.append("Recent conversation:\n" + "\n".join(turns))

    router_signals = (
        f"Router signals:\n"
        f"  user_intent: {inp.user_intent}\n"
        f"  needs_decomposition: {inp.needs_decomposition}\n"
        f"  entities: {[e.name for e in inp.router_entities]}"
    )
    parts.append(router_signals)
    parts.append(f"User query: {inp.user_query_raw}")

    return [
        ChatMessage(role=Role.system, content=system),
        ChatMessage(role=Role.user, content="\n\n".join(parts)),
    ]


def _parse_transformer_response(raw: str) -> tuple[TransformedQuery | None, str | None]:
    """Returns (TransformedQuery, None) on success, (None, error_msg) on failure."""
    text = (raw or "").strip()
    if not text:
        return None, "Empty response"
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}"
    # Strip runtime-only fields before validation so the LLM output validates cleanly
    data.pop("fallback", None)
    data.pop("decomposition_overridden", None)
    for sub in data.get("sub_queries", []):
        sub.pop("entity_match_quality", None)
    try:
        return TransformedQuery.model_validate(data), None
    except ValidationError as e:
        return None, f"Schema validation failed: {e}"


def _resolve_focus_entity(focus: str, known: list[str]) -> tuple[str | None, str]:
    """Exact-match-or-drop. Returns (resolved_name, match_quality)."""
    if focus in known:
        return focus, "exact"
    return None, "dropped"


def _normalize_transformed(
    raw: TransformedQuery,
    inp: TransformerInput,
    known_entities: list[str],
) -> TransformedQuery:
    """Deterministic post-LLM corrections. Returns fallback on fatal issues."""
    # Empty-string guard
    if not raw.semantic_query.strip() or not raw.keyword_query.strip():
        logger.warning("transformer_empty_rewrite")
        return _fallback(inp.user_query_raw)

    # Decomposition enforcement
    sub_queries = raw.sub_queries
    if not inp.needs_decomposition and sub_queries:
        logger.info("sub_queries_stripped", extra={"count": len(sub_queries)})
        sub_queries = []

    # Sub-query cap
    if len(sub_queries) > 5:
        logger.warning("sub_queries_capped", extra={"original": len(sub_queries)})
        sub_queries = sub_queries[:5]

    # Focus entity resolution — drop unresolved sub-queries
    resolved_subs: list[SubQuery] = []
    for sub in sub_queries:
        name, quality = _resolve_focus_entity(sub.focus_entity, known_entities)
        if quality == "dropped":
            logger.warning(
                "sub_query_dropped_unresolved_entity",
                extra={"focus_entity": sub.focus_entity},
            )
            continue
        resolved_subs.append(sub.model_copy(update={"entity_match_quality": quality}))

    return raw.model_copy(update={"sub_queries": resolved_subs})


async def transform_query(
    inp: TransformerInput,
    *,
    llm_router: LLMRouter | None = None,
    session: AsyncSession | None = None,
    parent_request_id: UUID | None = None,
    conversation_id: UUID | None = None,
    user_id: UUID | None = None,
) -> TransformedQuery:
    """Transform a raw user query into asymmetric semantic + keyword rewrites.

    Degrades gracefully: on any error returns fallback with raw query repeated.
    When session/parent_request_id/conversation_id are provided, each LLM call
    is logged to llm_requests as a sub-request for cost/token aggregation.
    """
    router = llm_router if llm_router is not None else get_router()
    model_id = get_query_transformer_model()
    cfg = get_query_transformer_config()
    should_log_subrequest = (
        session is not None and parent_request_id is not None and conversation_id is not None
    )
    request_params = {
        "temperature": cfg["temperature"],
        "max_tokens": int(cfg["max_tokens"]),
    }

    try:
        llm = router.get(model_id)
    except Exception:
        logger.warning("transform_query_model_unavailable", extra={"model": model_id})
        return _fallback(inp.user_query_raw)

    try:
        prompt = get_prompt_loader().load("query_transformer", "v2")
        system = get_prompt_renderer()._render_template(prompt.template, {})
    except Exception:
        logger.warning("transform_query_prompt_missing")
        return _fallback(inp.user_query_raw)

    response_format = _transformer_response_format()
    messages = _build_messages(inp, system)
    known_entities = inp.known_entity_names

    output: TransformedQuery | None = None
    for attempt in range(2):
        try:
            resp = await llm.complete(
                messages=messages,
                temperature=cfg["temperature"],
                max_tokens=int(cfg["max_tokens"]),
                response_format=response_format,
            )
        except Exception as e:
            logger.exception("transform_query_llm_error", extra={"error": str(e)})
            return _fallback(inp.user_query_raw)

        if should_log_subrequest:
            await LLMRequestRepository(session).create_subrequest(  # type: ignore[arg-type]
                parent_request_id=parent_request_id,  # type: ignore[arg-type]
                conversation_id=conversation_id,  # type: ignore[arg-type]
                user_id=user_id,
                provider=llm.provider,
                model=model_id,
                request_type="query_transformer",
                request_params=request_params,
                status="completed",
                **stats_to_request_kwargs(resp.stats),
            )

        raw_text = resp.text or ""
        result, error = _parse_transformer_response(raw_text)
        if result is not None:
            output = result
            break

        logger.warning(
            "transform_query_parse_failed attempt=%d error=%s raw=%r",
            attempt,
            error,
            raw_text[:300],
        )
        if attempt == 0:
            retry_content = (
                f"User query: {inp.user_query_raw}\n\n"
                f"Previous response: {raw_text[:300]}\n"
                f"Error: {error}\n\n"
                "Fix the error. Return ONLY valid JSON matching the required schema."
            )
            messages = [
                ChatMessage(role=Role.system, content=system),
                ChatMessage(role=Role.user, content=retry_content),
            ]

    if output is None:
        return _fallback(inp.user_query_raw)

    normalized = _normalize_transformed(output, inp, known_entities)
    logger.info(
        "transform_query_done fallback=%s sub_queries=%d",
        normalized.fallback,
        len(normalized.sub_queries),
    )
    return normalized
