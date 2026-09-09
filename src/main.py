from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from src.api.exceptions import llm_error_handler
from src.api.logging import configure_logging, request_logging_middleware
from src.api.routers import get_routers
from src.db import init_db, shutdown_db
from src.observability import langfuse as lf_client
from src.observability.metrics import open_stream_count
from src.redis_client import (
    close_redis_client,
    create_redis_app_client,
    create_redis_broker_client,
)
from src.services.llm_router import get_router
from src.services.llm_runtime.exceptions import LLMError
from src.utils.config import (
    get_cors_origins,
    get_max_open_streams,
    get_readiness_redis_timeout_seconds,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifecycle: startup and shutdown."""
    # Startup
    await init_db()
    app.state.llm_router = get_router()
    app.state.redis = await create_redis_app_client()
    # Separate instance from redis-app: the Celery queue lives on redis-broker, and admission
    # control reads its depth.
    app.state.redis_broker = await create_redis_broker_client()
    lf_client.initialize()
    logger.info("app.started")

    yield

    # Shutdown
    logger.info("app.shutting_down")
    lf_client.flush()
    await app.state.llm_router.close()
    await close_redis_client(app.state.redis)
    await close_redis_client(app.state.redis_broker)
    await shutdown_db()
    logger.info("db.shutdown")
    logger.info("app.stopped")


def create_app() -> FastAPI:
    configure_logging()

    app = FastAPI(title="AI Financial Copilot API", lifespan=lifespan)

    # CORS middleware (must be added before other middleware for preflight handling)
    cors_origins = get_cors_origins()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "Retry-After"],
    )
    logger.info("cors.configured", extra={"origins": cors_origins})

    app.middleware("http")(request_logging_middleware)

    # Expose Prometheus metrics (scraped by Prometheus locally / Operator in K8s)
    app.mount("/metrics", make_asgi_app())

    # Register global exception handler for LLM errors
    app.add_exception_handler(LLMError, llm_error_handler)

    for router in get_routers():
        app.include_router(router)

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> JSONResponse:
        """Liveness only: "is the event loop still turning". Deliberately static — failing
        this kills the container and every SSE stream on it, so it must never depend on a
        dependency that can blip."""
        return JSONResponse({"status": "ok"})

    @app.get("/readyz", include_in_schema=False)
    async def readyz() -> JSONResponse:
        """Readiness: "should traffic come to me right now". Failing is safe and reversible —
        the pod leaves the endpoint list and rejoins on recovery.

        Both checks are deliberately pod-local. Redis is shared, so it is pinged with a short
        timeout and nothing more; anything stricter on a shared dependency would take every
        replica NotReady at once and turn a degraded system into an outage.
        """
        try:
            await asyncio.wait_for(
                app.state.redis.ping(), timeout=get_readiness_redis_timeout_seconds()
            )
        except Exception:
            logger.warning("readyz.redis_unavailable")
            return JSONResponse({"status": "redis_unavailable"}, status_code=503)

        open_streams = open_stream_count()
        max_streams = get_max_open_streams()
        if open_streams >= max_streams:
            logger.warning(
                "readyz.at_capacity",
                extra={"open_streams": open_streams, "max_open_streams": max_streams},
            )
            return JSONResponse(
                {"status": "at_capacity", "open_streams": open_streams}, status_code=503
            )

        return JSONResponse({"status": "ok", "open_streams": open_streams})

    return app


app = create_app()
