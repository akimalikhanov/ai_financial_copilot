from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import aclosing

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from src.api.deps import LLMRouterDep
from src.api.exceptions import _error_to_schema, _sse_event
from src.api.logging import set_request_error, set_request_meta
from src.schemas import chat as schemas
from src.services.llm_adapters.base_adapter import (
    ChatMessage as AdapterChatMessage,
)
from src.services.llm_adapters.base_adapter import (
    LLMResponse as AdapterResponse,
)
from src.services.llm_adapters.base_adapter import (
    LLMResponseStats as AdapterResponseStats,
)
from src.services.llm_adapters.base_adapter import (
    LLMStreamChunk as AdapterStreamChunk,
)
from src.services.llm_adapters.base_adapter import (
    Role as AdapterRole,
)
from src.services.llm_runtime.exceptions import LLMError

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


@router.post("", response_model=schemas.LLMResponse, response_model_exclude_none=True)
async def chat(
    req: schemas.ChatRequest,
    llm_router: LLMRouterDep,
) -> schemas.LLMResponse:
    messages = _to_adapter_messages(req.messages)
    llm = llm_router.get(req.model)
    # Enrich request log with LLM metadata
    set_request_meta(provider=llm.provider, model_id=llm.model_id)
    response = await llm.complete(
        messages=messages,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        **req.extra_params,
    )
    return _response_to_schema(response)


@router.post("/stream")
async def chat_stream(
    req: schemas.ChatRequest,
    llm_router: LLMRouterDep,
) -> StreamingResponse:
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }

    messages = _to_adapter_messages(req.messages)
    llm = llm_router.get(req.model)
    # Enrich request log with LLM metadata
    set_request_meta(provider=llm.provider, model_id=llm.model_id)
    stream = llm.stream(
        messages=messages,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        **req.extra_params,
    )

    async def event_stream() -> AsyncGenerator[str, None]:
        # Use aclosing to ensure stream cleanup on client disconnect or errors
        async with aclosing(stream) as safe_stream:
            try:
                async for chunk in safe_stream:
                    event = "usage" if chunk.is_final else "delta"
                    payload = _stream_chunk_to_schema(chunk).model_dump(exclude_none=True)
                    yield _sse_event(event, payload)
            except LLMError as exc:
                # Errors during streaming are handled here (can't be caught by exception handler)
                set_request_error(error_type=type(exc).__name__, error_msg=str(exc))
                error_payload = _error_to_schema(exc).model_dump(exclude_none=True)
                yield _sse_event("error", error_payload)
            except Exception as exc:
                # Catch-all for unexpected errors during streaming
                set_request_error(error_type="InternalServerError", error_msg=str(exc))
                error_payload = schemas.ErrorResponse(
                    error_type="InternalServerError",
                    message="Internal server error",
                ).model_dump(exclude_none=True)
                yield _sse_event("error", error_payload)

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)
