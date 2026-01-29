from __future__ import annotations

import logging
from time import perf_counter
from uuid import uuid4

from fastapi import Request
from starlette.responses import Response

from src.api.logging import reset_request_id, set_request_id

REQUEST_ID_HEADER = "x-request-id"

logger = logging.getLogger(__name__)


async def request_id_middleware(request: Request, call_next) -> Response:
    request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid4())
    token = set_request_id(request_id)
    request.state.request_id = request_id

    start = perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (perf_counter() - start) * 1000
        logger.exception(
            "request.failed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": 500,
                "duration_ms": round(duration_ms, 3),
            },
        )
        reset_request_id(token)
        raise

    response.headers[REQUEST_ID_HEADER] = request_id

    duration_ms = (perf_counter() - start) * 1000
    logger.info(
        "request.completed",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(duration_ms, 3),
        },
    )
    reset_request_id(token)
    return response
