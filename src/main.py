from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.logging import configure_logging
from src.api.middleware import request_id_middleware
from src.api.routers import get_routers
from src.services.llm_router import get_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
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
    app.middleware("http")(request_id_middleware)

    for router in get_routers():
        app.include_router(router)

    return app


app = create_app()
