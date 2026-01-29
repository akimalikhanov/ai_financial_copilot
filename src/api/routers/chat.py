from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from src.api.deps import get_llm_router
from src.schemas import chat as schemas
from src.services.llm_adapters.base_adapter import (
    ChatMessage as AdapterChatMessage,
    LLMResponse as AdapterResponse,
    LLMResponseStats as AdapterResponseStats,
    LLMStreamChunk as AdapterStreamChunk,
    Role as AdapterRole,
)
from src.services.llm_router import LLMRouter
from src.services.llm_runtime.exceptions import LLMError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/chat", tags=["chat"])


def _to_adapter_messages(messages: list[schemas.ChatMessage]) -> list[AdapterChatMessage]:
    return [
        AdapterChatMessage(
            role=AdapterRole(message.role),
            content=message.content,
            name=message.name,
            tool_call_id=message.tool_call_id,
        )
        for message in messages
    ]


def _stats_to_schema(stats: AdapterResponseStats | None) -> schemas.LLMResponseStats | None:
    if stats is None:
        return None
    return schemas.LLMResponseStats(
        input_tokens=stats.input_tokens,
        cached_input_tokens=stats.cached_input_tokens,
        output_tokens=stats.output_tokens,
        reasoning_tokens=stats.reasoning_tokens,
        total_tokens=stats.total_tokens,
        latency_ms=stats.latency_ms,
        ttft_ms=stats.ttft_ms,
        tps=stats.tps,
        cost_usd=stats.cost_usd,
    )


def _response_to_schema(response: AdapterResponse) -> schemas.LLMResponse:
    return schemas.LLMResponse(
        text=response.text,
        stats=_stats_to_schema(response.stats),
        raw=None,
    )


def _stream_chunk_to_schema(chunk: AdapterStreamChunk) -> schemas.LLMStreamChunk:
    return schemas.LLMStreamChunk(
        text=chunk.text,
        is_final=chunk.is_final,
        stats=_stats_to_schema(chunk.stats),
        raw=None,
    )


def _error_to_schema(error: LLMError) -> schemas.ErrorResponse:
    payload = error.to_dict()
    return schemas.ErrorResponse(
        error_type=payload.get("error_type", type(error).__name__),
        message=payload.get("message", str(error)),
        internal_message=payload.get("internal_message"),
        user_message=payload.get("user_message"),
        provider=payload.get("provider"),
        model=payload.get("model"),
        is_retryable=payload.get("is_retryable"),
        status_code=payload.get("status_code"),
        error_code=payload.get("error_code"),
        original_error_message=str(error.original_error) if error.original_error else None,
    )


def _error_response(error: LLMError) -> JSONResponse:
    payload = _error_to_schema(error)
    status_code = error.status_code or 500
    content = payload.model_dump(exclude_none=True)
    content.setdefault("status_code", status_code)

    headers: dict[str, str] = {}
    if error.retry_info and error.retry_info.retry_after_seconds is not None:
        headers["Retry-After"] = str(error.retry_info.retry_after_seconds)

    return JSONResponse(status_code=status_code, content=content, headers=headers)


def _sse_event(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=True)}\n\n"


@router.post("", response_model=schemas.LLMResponse, response_model_exclude_none=True)
async def chat(
    req: schemas.ChatRequest,
    llm_router: LLMRouter = Depends(get_llm_router),
) -> schemas.LLMResponse:
    try:
        messages = _to_adapter_messages(req.messages)
        llm = llm_router.get(req.model)
        response = await llm.complete(
            messages=messages,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            **req.extra_params,
        )
    except LLMError as exc:
        return _error_response(exc)
    return _response_to_schema(response)


@router.post("/stream")
async def chat_stream(
    req: schemas.ChatRequest,
    request: Request,
    llm_router: LLMRouter = Depends(get_llm_router),
) -> StreamingResponse:
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }

    try:
        messages = _to_adapter_messages(req.messages)
        llm = llm_router.get(req.model)
        stream = llm.stream(
            messages=messages,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            **req.extra_params,
        )
    except LLMError as exc:
        error_payload = _error_to_schema(exc).model_dump(exclude_none=True)

        async def error_stream() -> AsyncIterator[str]:
            yield _sse_event("error", error_payload)

        return StreamingResponse(error_stream(), media_type="text/event-stream", headers=headers)

    async def event_stream() -> AsyncIterator[str]:
        try:
            async for chunk in stream:
                event = "usage" if chunk.is_final else "delta"
                payload = _stream_chunk_to_schema(chunk).model_dump(exclude_none=True)
                yield _sse_event(event, payload)
        except LLMError as exc:
            error_payload = _error_to_schema(exc).model_dump(exclude_none=True)
            yield _sse_event("error", error_payload)
        except Exception:
            logger.exception("chat.stream.failed", extra={"path": request.url.path})
            error_payload = schemas.ErrorResponse(
                error_type="InternalServerError",
                message="Internal server error",
            ).model_dump(exclude_none=True)
            yield _sse_event("error", error_payload)

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)
