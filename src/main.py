from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.exceptions import llm_error_handler
from src.api.logging import configure_logging, request_logging_middleware
from src.api.routers import get_routers
from src.services.llm_router import get_router
from src.services.llm_runtime.exceptions import LLMError

logger = logging.getLogger(__name__)

def _get_cors_origins() -> list[str]:
    """
    Parse CORS_ALLOWED_ORIGINS from environment.

    CORS_ALLOWED_ORIGINS: Comma-separated list of allowed origins (required).
    Examples:
      - Development: "http://localhost:3000,http://127.0.0.1:3000"
      - Production:  "https://app.example.com,https://www.example.com"
      - Allow all (NOT recommended for production): "*"

    Raises:
        RuntimeError: If CORS_ALLOWED_ORIGINS is not set.
    """
    raw = os.getenv("CORS_ALLOWED_ORIGINS")
    if not raw:
        raise RuntimeError(
            "CORS_ALLOWED_ORIGINS environment variable is required. "
            "Set it to a comma-separated list of allowed origins "
            "(e.g., 'http://localhost:3000' for dev, or your production domain)."
        )
    if raw == "*":
        return ["*"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifecycle: startup and shutdown."""
    # Startup
    app.state.llm_router = get_router()
    logger.info("app.started")

    yield

    # Shutdown
    logger.info("app.shutting_down")
    await app.state.llm_router.close()
    logger.info("app.stopped")


def create_app() -> FastAPI:
    configure_logging()

    app = FastAPI(title="AI Financial Copilot API", lifespan=lifespan)

    # CORS middleware (must be added before other middleware for preflight handling)
    cors_origins = _get_cors_origins()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
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
