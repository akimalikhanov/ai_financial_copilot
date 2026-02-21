from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.exceptions import llm_error_handler
from src.api.logging import configure_logging, request_logging_middleware
from src.api.routers import get_routers
from src.db import init_db, shutdown_db
from src.redis_client import (
    close_redis_client,
    create_redis_app_client,
    create_redis_broker_client,
)
from src.services.llm_router import get_router
from src.services.llm_runtime.exceptions import LLMError
from src.utils.config import get_cors_origins

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifecycle: startup and shutdown."""
    # Startup
    await init_db()
    app.state.llm_router = get_router()
    app.state.redis = await create_redis_app_client()
    app.state.redis_broker = await create_redis_broker_client()
    logger.info("app.started")

    yield

    # Shutdown
    logger.info("app.shutting_down")
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

    # Register global exception handler for LLM errors
    app.add_exception_handler(LLMError, llm_error_handler)

    for router in get_routers():
        app.include_router(router)

    return app


app = create_app()
