"""
Query preprocessing and lightweight routing for the RAG pipeline.

Performs minimal deterministic preprocessing and routes user turns into:
- direct_answer: greetings, thanks, help, capability questions, out-of-scope (main LLM handles guardrails)
- retrieve: corpus-grounded questions, report/filing/company/metric queries
"""

from __future__ import annotations

import json
import logging
import re

from src.schemas.retrieval import ProcessedQuery, RouterOutput
from src.services.llm_adapters.base_adapter import ChatMessage, Role
from src.services.llm_router import LLMRouter, get_router
from src.services.prompts.prompt_loader import get_prompt_loader
from src.utils.config import get_query_router_model

logger = logging.getLogger(__name__)


def _router_schema_for_openai() -> dict:
    """Schema with additionalProperties: false for OpenAI strict mode. required must include all properties."""
    schema = dict(RouterOutput.model_json_schema())
    schema["additionalProperties"] = False
    schema["required"] = list(schema["properties"].keys())
    return schema


_ROUTER_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "query_router",
        "schema": _router_schema_for_openai(),
        "strict": True,
    },
}


def _normalize(raw: str) -> str:
    """Strip and collapse whitespace. Raises ValueError if empty."""
    cleaned = raw.strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned:
        raise ValueError("Query cannot be empty")
    return cleaned


def _get_router_system_prompt() -> str:
    """Load query router system prompt from prompts dir."""
    loader = get_prompt_loader()
    template = loader.load("query_router", "v1")
    return template.template


def _parse_router_response(
    text: str, normalized_text: str
) -> tuple[ProcessedQuery | None, str | None]:
    """Parse and validate LLM JSON output into RouterOutput. Returns (result, error_msg)."""
    text = (text or "").strip()
    if not text:
        return None, "Empty response"
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}"
    try:
        out = RouterOutput.model_validate(data)
    except Exception as e:
        return None, f"Schema validation failed: {e}"
    return (
        ProcessedQuery(
            normalized_text=normalized_text,
            route=out.route,
            user_intent=out.user_intent[:200],
            reason=out.reason[:500] if out.reason else None,
        ),
        None,
    )


async def _route_with_llm(
    normalized_text: str,
    *,
    router: LLMRouter | None = None,
) -> ProcessedQuery | None:
    """Call LLM for structured routing. Returns None on failure. Retries once on schema validation failure."""
    llm_router = router if router is not None else get_router()
    model_id = get_query_router_model()

    try:
        llm = llm_router.get(model_id)
    except Exception:
        logger.warning("query_router_model_unavailable", extra={"model": model_id})
        return None

    system = _get_router_system_prompt()
    user_content = f"User query: {normalized_text}"
    messages: list[ChatMessage] = [
        ChatMessage(role=Role.system, content=system),
        ChatMessage(role=Role.user, content=user_content),
    ]

    for attempt in range(2):
        try:
            resp = await llm.complete(
                messages=messages,
                temperature=0,
                max_tokens=500,
                response_format=_ROUTER_RESPONSE_FORMAT,
            )
        except Exception as e:
            logger.exception("query_router_llm_error", extra={"error": str(e)})
            return None

        raw_text = resp.text or ""
        result, error_msg = _parse_router_response(raw_text, normalized_text)
        if result is not None:
            logger.info(
                "query_router_result route=%s intent=%s reason=%s",
                result.route,
                result.user_intent,
                result.reason,
            )
            return result
        logger.warning(
            "query_router_parse_failed raw_response=%r error=%s",
            raw_text[:500],
            error_msg,
        )
        if attempt == 0:
            retry_content = (
                f"{user_content}\n\n"
                f"Previous response: {resp.text[:500] if resp.text else '(empty)'}\n"
                f"Error: {error_msg}\n\n"
                "Fix error. Return only valid JSON matching the required schema."
            )
            messages = [
                ChatMessage(role=Role.system, content=system),
                ChatMessage(role=Role.user, content=retry_content),
            ]
            logger.debug("query_router_retry", extra={"attempt": 1, "error": error_msg})

    return None


async def process_query(raw_query: str, *, router: LLMRouter | None = None) -> ProcessedQuery:
    """
    Preprocess and route the user query.

    - Preprocessing: strip, collapse whitespace; reject empty with ValueError.
    - Routing: lightweight structured LLM step; on failure, default to retrieve.
    - router: Use pre-initialized router from app.state.llm_router when available.
    """
    normalized = _normalize(raw_query)

    result = await _route_with_llm(normalized, router=router)
    if result is not None:
        return result

    return ProcessedQuery(
        normalized_text=normalized,
        route="retrieve",
        user_intent="fallback",
        reason="router_fallback",
    )
