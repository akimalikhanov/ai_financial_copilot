from __future__ import annotations

import json
import logging
from typing import Literal
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from src.repository.llm_request_repository import LLMRequestRepository, stats_to_request_kwargs
from src.schemas.query_transform import ScopeDocSummary, TransformedQuery, TransformerInput
from src.services.llm_adapters.base_adapter import ChatMessage, LLMResponseStats, Role
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


def _rewriter_response_format() -> dict:
    schema = TransformedQuery.model_json_schema()
    schema.get("properties", {}).pop("fallback", None)
    if "required" in schema:
        schema["required"] = [f for f in schema["required"] if f != "fallback"]
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

    if inp.conversation_history:
        turns = []
        for turn in inp.conversation_history:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role == "assistant":
                content = _truncate_to_tokens(content, TRANSFORMER_CONV_HISTORY_TOKENS)
            turns.append(f"{role}: {content}")
        parts.append("Recent conversation:\n" + "\n".join(turns))

    parts.append(f"Router signals:\n  user_intent: {inp.user_intent}")
    parts.append(f"User query: {inp.user_query_raw}")

    return [
        ChatMessage(role=Role.system, content=system),
        ChatMessage(role=Role.user, content="\n\n".join(parts)),
    ]


def _parse_response(raw: str) -> tuple[TransformedQuery | None, str | None]:
    text = (raw or "").strip()
    if not text:
        return None, "Empty response"
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}"
    data.pop("fallback", None)
    try:
        return TransformedQuery.model_validate(data), None
    except ValidationError as e:
        return None, f"Schema validation failed: {e}"


def _normalize(raw: TransformedQuery, user_query_raw: str) -> TransformedQuery:
    if not raw.semantic_query.strip() or not raw.keyword_query.strip():
        logger.warning("rewrite_query_empty_rewrite")
        return _fallback(user_query_raw)
    return raw


def _modes_for(search_mode: Literal["hybrid", "vector", "keyword"]) -> dict:
    """Return prompt hints for which queries are needed given the search mode."""
    return {
        "hybrid": {},
        "vector": {"skip_keyword": True},
        "keyword": {"skip_semantic_embed": True},
    }[search_mode]


async def rewrite_query(
    raw_query: str,
    *,
    scope_docs: list[ScopeDocSummary] | None = None,
    conversation_history: list[dict] | None = None,
    user_intent: str = "retrieval",
    llm_router: LLMRouter | None = None,
    session: AsyncSession | None = None,
    parent_request_id: UUID | None = None,
    conversation_id: UUID | None = None,
    user_id: UUID | None = None,
    extra_request_params: dict | None = None,
) -> tuple[TransformedQuery, LLMResponseStats | None]:
    """Rewrite a raw query into semantic + keyword forms for hybrid RAG.

    Replaces transform_query(). No decomposition — the agent loop handles multi-entity
    by issuing one rewrite_query call per search_documents tool call.
    Degrades gracefully: any error returns fallback with raw query repeated.
    """
    router = llm_router if llm_router is not None else get_router()
    model_id = get_query_transformer_model()
    cfg = get_query_transformer_config()
    should_log_subrequest = (
        session is not None and parent_request_id is not None and conversation_id is not None
    )
    request_params: dict = {
        "temperature": cfg["temperature"],
        "max_tokens": int(cfg["max_tokens"]),
        **(extra_request_params or {}),
    }

    try:
        llm = router.get(model_id)
    except Exception:
        logger.warning("rewrite_query_model_unavailable", extra={"model": model_id})
        return _fallback(raw_query), None

    try:
        prompt = get_prompt_loader().load("query_transformer", "v2")
        system = get_prompt_renderer()._render_template(prompt.template, {})
    except Exception:
        logger.warning("rewrite_query_prompt_missing")
        return _fallback(raw_query), None

    inp = TransformerInput(
        user_query_raw=raw_query,
        conversation_history=conversation_history or [],
        user_intent=user_intent,
        scope_docs=scope_docs or [],
    )
    response_format = _rewriter_response_format()
    messages = _build_messages(inp, system)

    output: TransformedQuery | None = None
    last_stats: LLMResponseStats | None = None
    for attempt in range(2):
        try:
            resp = await llm.complete(
                messages=messages,
                _lf_name="rewrite_query",
                temperature=cfg["temperature"],
                max_tokens=int(cfg["max_tokens"]),
                response_format=response_format,
            )
        except Exception as e:
            logger.exception("rewrite_query_llm_error", extra={"error": str(e)})
            return _fallback(raw_query), last_stats

        last_stats = resp.stats

        if should_log_subrequest:
            await LLMRequestRepository(session).create_subrequest(  # type: ignore[arg-type]
                parent_request_id=parent_request_id,  # type: ignore[arg-type]
                conversation_id=conversation_id,  # type: ignore[arg-type]
                user_id=user_id,
                provider=llm.provider,
                model=model_id,
                request_type="rewrite_query",
                request_params=request_params,
                status="completed",
                **stats_to_request_kwargs(resp.stats),
            )

        result, error = _parse_response(resp.text or "")
        if result is not None:
            output = result
            break

        logger.warning(
            "rewrite_query_parse_failed attempt=%d error=%s raw=%r",
            attempt,
            error,
            (resp.text or "")[:300],
        )
        if attempt == 0:
            retry_content = (
                f"User query: {raw_query}\n\n"
                f"Previous response: {(resp.text or '')[:300]}\n"
                f"Error: {error}\n\n"
                "Fix the error. Return ONLY valid JSON matching the required schema."
            )
            messages = [
                ChatMessage(role=Role.system, content=system),
                ChatMessage(role=Role.user, content=retry_content),
            ]

    if output is None:
        return _fallback(raw_query), last_stats

    normalized = _normalize(output, raw_query)
    logger.info(
        "rewrite_query_done fallback=%s",
        normalized.fallback,
        extra={"model": model_id, "provider": llm.provider},
    )
    return normalized, last_stats
