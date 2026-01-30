from __future__ import annotations

import json
import logging
import os
import sys
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import UTC, datetime
from time import perf_counter
from typing import Any
from uuid import uuid4

from fastapi import Request
from starlette.responses import Response

# ---------------------------------------------------------------------------
# Request Context (ContextVar-based, accumulates data during request lifecycle)
# ---------------------------------------------------------------------------

REQUEST_ID_HEADER = "x-request-id"


@dataclass
class RequestContext:
    """Accumulated request metadata for unified logging."""

    request_id: str
    method: str = ""
    path: str = ""
    # Populated by route handlers
    provider: str | None = None
    model_id: str | None = None
    # Populated at end of request
    http_status: int | None = None
    status: str = "success"  # "success" | "error"
    error_type: str | None = None
    error_msg: str | None = None
    # Timing
    start_time: float = field(default_factory=perf_counter)
    duration_ms: float | None = None
    # Extra fields from route handlers
    extra: dict[str, Any] = field(default_factory=dict)


_request_ctx: ContextVar[RequestContext | None] = ContextVar("request_ctx", default=None)


def get_request_context() -> RequestContext | None:
    """Get current request context (if inside a request)."""
    return _request_ctx.get()


def get_request_id() -> str | None:
    """Get current request ID (convenience function for backward compat)."""
    ctx = _request_ctx.get()
    return ctx.request_id if ctx else None


def set_request_meta(
    *,
    provider: str | None = None,
    model_id: str | None = None,
    **extra: Any,
) -> None:
    """Set LLM-related metadata on the current request context (called from routes)."""
    ctx = _request_ctx.get()
    if ctx is None:
        return
    if provider is not None:
        ctx.provider = provider
    if model_id is not None:
        ctx.model_id = model_id
    if extra:
        ctx.extra.update(extra)


def set_request_error(
    *,
    error_type: str | None = None,
    error_msg: str | None = None,
) -> None:
    """Mark current request as errored (called from routes or exception handlers)."""
    ctx = _request_ctx.get()
    if ctx is None:
        return
    ctx.status = "error"
    if error_type is not None:
        ctx.error_type = error_type
    if error_msg is not None:
        ctx.error_msg = error_msg


# ---------------------------------------------------------------------------
# JSON Formatter
# ---------------------------------------------------------------------------

_STANDARD_LOG_RECORD_ATTRS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "message",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        # Use record.created for consistent timestamp (already captured at log time)
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add request context if available
        ctx = _request_ctx.get()
        if ctx is not None:
            payload["request_id"] = ctx.request_id

        # Add extra fields from log record
        for key, value in record.__dict__.items():
            if key in _STANDARD_LOG_RECORD_ATTRS or key.startswith("_"):
                continue
            if key in payload:
                continue
            payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=True)


# ---------------------------------------------------------------------------
# Unified Request Logging Middleware
# ---------------------------------------------------------------------------

_request_logger = logging.getLogger("api.request")


async def request_logging_middleware(request: Request, call_next) -> Response:
    """
    Unified request middleware that:
    1. Sets up request context with ID and timing
    2. Calls the route handler
    3. Emits a single JSON log with all accumulated metadata
    """
    request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid4())

    ctx = RequestContext(
        request_id=request_id,
        method=request.method,
        path=request.url.path,
    )
    token: Token[RequestContext | None] = _request_ctx.set(ctx)
    request.state.request_id = request_id

    try:
        response = await call_next(request)
        ctx.http_status = response.status_code
        # Infer error status from HTTP code if not already set
        if response.status_code >= 400 and ctx.status == "success":
            ctx.status = "error"
    except Exception as exc:
        ctx.http_status = 500
        ctx.status = "error"
        ctx.error_type = type(exc).__name__
        ctx.error_msg = str(exc)
        _log_request(ctx, exc_info=True)
        _request_ctx.reset(token)
        raise

    response.headers[REQUEST_ID_HEADER] = request_id
    _log_request(ctx)
    _request_ctx.reset(token)
    return response


def _log_request(ctx: RequestContext, exc_info: bool = False) -> None:
    """Emit a single unified log entry for the request."""
    ctx.duration_ms = round((perf_counter() - ctx.start_time) * 1000, 3)

    extra: dict[str, Any] = {
        "method": ctx.method,
        "path": ctx.path,
        "http_status": ctx.http_status,
        "status": ctx.status,
        "duration_ms": ctx.duration_ms,
    }

    # Add optional fields only if present
    if ctx.provider:
        extra["provider"] = ctx.provider
    if ctx.model_id:
        extra["model_id"] = ctx.model_id
    if ctx.error_type:
        extra["error_type"] = ctx.error_type
    if ctx.error_msg:
        extra["error_msg"] = ctx.error_msg
    if ctx.extra:
        extra.update(ctx.extra)

    level = logging.ERROR if ctx.status == "error" else logging.INFO
    _request_logger.log(
        level,
        "request.completed",
        extra=extra,
        exc_info=exc_info,
    )


# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------


def configure_logging(level: str | int | None = None) -> None:
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")

    # Create shared JSON handler
    json_handler = logging.StreamHandler(sys.stdout)
    json_handler.setFormatter(JsonFormatter())

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(json_handler)
    root_logger.setLevel(level)

    # Hijack uvicorn loggers to use our JSON format
    for uvicorn_logger_name in ("uvicorn", "uvicorn.error"):
        uvicorn_logger = logging.getLogger(uvicorn_logger_name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.addHandler(json_handler)
        uvicorn_logger.propagate = False  # Don't double-log to root

    # Silence noisy third-party loggers (we have our own unified request logging)
    for logger_name in (
        "uvicorn.access",  # We have our own request logging
        "httpx",
        "httpcore",
        "openai",
        "google_genai",
        "google.genai",
    ):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
