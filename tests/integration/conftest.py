"""Integration test fixtures for full flow (API → queue → worker → DB + SSE)."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.api.routers import get_routers
from src.db import init_db, shutdown_db
from src.redis_client import close_redis_client, create_redis_client
from src.services.llm_adapters.base_adapter import LLMStreamChunk
from src.services.llm_router import LLMRouter, RoutedLLM

# Distinctive mock response to verify mock is used (avoids real LLM calls)
MOCK_RESPONSE = "[INTEGRATION-TEST-MOCK-RESPONSE]"


class MockStreamingLLM:
    """Mock LLM that streams a fixed response for integration tests."""

    def __init__(self, response_text: str = MOCK_RESPONSE) -> None:
        self.provider = "mock"
        self.model_id = "mock-model"
        self._response_text = response_text

    async def close(self) -> None:
        pass

    def stream(self, *_args: Any, **_kwargs: Any) -> AsyncGenerator[LLMStreamChunk, None]:
        async def _gen() -> AsyncGenerator[LLMStreamChunk, None]:
            # Emit one delta, then final chunk (matches worker expectation)
            yield LLMStreamChunk(text=self._response_text, is_final=False)
            yield LLMStreamChunk(text="", is_final=True, stats=None)

        return _gen()


def _create_mock_router(response_text: str = MOCK_RESPONSE) -> LLMRouter:
    """Create LLMRouter with mock streaming model."""
    mock_llm = MockStreamingLLM(response_text=response_text)
    routed = RoutedLLM(
        adapter=mock_llm,  # type: ignore[arg-type]
        provider="mock",
        model_id="gpt-4o-mini",
        default_params={"temperature": 0.2, "max_tokens": 2000},
        default_stream=True,
        capabilities={},
    )
    config = {
        "defaults": {"stream": True, "params": {"temperature": 0.2, "max_tokens": 2000}},
        "models": [],
    }
    router = LLMRouter(config)
    router._models["gpt-4o-mini"] = routed
    return router


@asynccontextmanager
async def app_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan that uses mock LLM router."""
    await init_db()
    app.state.llm_router = _create_mock_router()
    app.state.redis = await create_redis_client()

    yield

    await app.state.llm_router.close()
    await close_redis_client(app.state.redis)
    await shutdown_db()


@pytest.fixture(autouse=True)
def _integration_redis_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use isolated Redis stream so real worker won't consume test jobs."""
    monkeypatch.setattr("src.redis_client.CHAT_QUEUE_STREAM", "chat:queue:integration-test")
    monkeypatch.setattr("src.workers.chat_worker.CHAT_QUEUE_STREAM", "chat:queue:integration-test")


@pytest.fixture(autouse=True)
def _patch_llm_router(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure get_router returns mock in integration tests (no real API calls)."""
    mock_router = _create_mock_router()
    monkeypatch.setattr("src.services.llm_router.get_router", lambda *_args, **_kwargs: mock_router)


@pytest.fixture(autouse=True)
def _jwt_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set JWT env for auth integration tests."""
    monkeypatch.setenv("JWT_SECRET_KEY", "integration-test-secret-key-min-32-chars")
    monkeypatch.setenv("JWT_ALGORITHM", "HS256")


@pytest.fixture
def integration_app() -> FastAPI:
    """FastAPI app with mock LLM for integration tests."""
    from fastapi import FastAPI

    app = FastAPI(title="Integration Test API", lifespan=app_lifespan)
    for router in get_routers():
        app.include_router(router)
    return app


@pytest.fixture
async def async_client(integration_app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP client for integration tests. Lifespan runs for test duration."""
    async with app_lifespan(integration_app):
        transport = ASGITransport(app=integration_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client
