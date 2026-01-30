from __future__ import annotations

import json
from typing import Any, cast

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.responses import Response

from src.api.logging import set_request_error
from src.schemas import chat as schemas
from src.services.llm_runtime.exceptions import LLMError


def _sse_event(event: str, data: dict[str, Any]) -> str:
    """Format Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=True)}\n\n"


def _error_to_schema(error: LLMError) -> schemas.ErrorResponse:
    """Convert LLMError to ErrorResponse schema."""
    payload = cast(dict[str, Any], error.to_dict(as_json=False))
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
    """Create JSONResponse from LLMError."""
    payload = _error_to_schema(error)
    status_code = error.status_code or 500
    content = payload.model_dump(exclude_none=True)
    content.setdefault("status_code", status_code)

    headers: dict[str, str] = {}
    if error.retry_info and error.retry_info.retry_after_seconds is not None:
        headers["Retry-After"] = str(error.retry_info.retry_after_seconds)

    return JSONResponse(status_code=status_code, content=content, headers=headers)


async def llm_error_handler(request: Request, exc: Exception) -> Response:
    """
    Global exception handler for LLMError.
    Handles both regular JSON responses and streaming responses.
    """
    # Type narrow to LLMError (this is safe because we only register this for LLMError)
    if not isinstance(exc, LLMError):
        raise TypeError(f"Expected LLMError, got {type(exc).__name__}")

    # Enrich request log with error info
    set_request_error(error_type=type(exc).__name__, error_msg=str(exc))

    # Check if this is a streaming endpoint
    if "/stream" in request.url.path:
        # For streaming endpoints, return an error stream
        error_payload = _error_to_schema(exc).model_dump(exclude_none=True)

        async def error_stream():
            yield _sse_event("error", error_payload)

        return StreamingResponse(
            error_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # For non-streaming endpoints, return JSON error response
    return _error_response(exc)
