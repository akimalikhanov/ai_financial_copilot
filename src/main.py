from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.exceptions import llm_error_handler
from src.api.logging import configure_logging, request_logging_middleware
from src.api.routers import get_routers
from src.services.llm_router import get_router
from src.services.llm_runtime.exceptions import LLMError

logger = logging.getLogger(__name__)


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
    app.middleware("http")(request_logging_middleware)

    # Register global exception handler for LLM errors
    app.add_exception_handler(LLMError, llm_error_handler)

    for router in get_routers():
        app.include_router(router)

    return app


app = create_app()
