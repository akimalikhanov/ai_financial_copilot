"""Integration test fixtures for full flow (API → queue → worker → DB + SSE)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.api.routers import get_routers
from src.db import init_db, shutdown_db
from src.redis_client import close_redis_client, create_redis_app_client
from src.services.llm_adapters.base_adapter import (
    AssistantTurnResult,
    LLMResponse,
    LLMStreamChunk,
    ToolCallRef,
)
from src.services.llm_router import LLMRouter, RoutedLLM

# Distinctive mock response to verify mock is used (avoids real LLM calls)
MOCK_RESPONSE = "[INTEGRATION-TEST-MOCK-RESPONSE]"

TOOL_MODEL_ID = "mock-tool-model"

RETRIEVAL_ROUTER_JSON = json.dumps(
    {
        "route": "retrieval",
        "entities": [],
        "user_intent": "financial question",
        "reasoning": "needs document lookup",
        "query_shape": "extraction",
    }
)

_REPORT_FINDINGS_TC = ToolCallRef(
    id="call_1",
    name="report_findings",
    arguments=json.dumps({"metric_requested": "revenue", "findings": []}),
)


class MockStreamingLLM:
    """Mock LLM answering the router's `.complete()` call and the synthesis `.stream()` call."""

    def __init__(
        self, response_text: str = MOCK_RESPONSE, router_json: str = RETRIEVAL_ROUTER_JSON
    ) -> None:
        self.provider = "mock"
        self.model_id = "mock-model"
        self._response_text = response_text
        self._router_json = router_json

    async def close(self) -> None:
        pass

    async def complete(self, *_args: Any, **_kwargs: Any) -> LLMResponse:
        return LLMResponse(text=self._router_json, stats=None)

    def stream(self, *_args: Any, **_kwargs: Any) -> AsyncGenerator[LLMStreamChunk, None]:
        async def _gen() -> AsyncGenerator[LLMStreamChunk, None]:
            # Emit one delta, then final chunk (matches worker expectation)
            yield LLMStreamChunk(text=self._response_text, is_final=False)
            yield LLMStreamChunk(text="", is_final=True, stats=None)

        return _gen()


class MockToolLLM:
    """Mock tool-calling LLM: finalizes immediately with report_findings (no search)."""

    def __init__(self) -> None:
        self.provider = "mock"
        self.model_id = TOOL_MODEL_ID

    async def close(self) -> None:
        pass

    async def complete_with_tools(self, *_args: Any, **_kwargs: Any) -> AssistantTurnResult:
        return AssistantTurnResult(text="", tool_calls=[_REPORT_FINDINGS_TC])


def create_agentic_router(
    chat_adapter: MockStreamingLLM | None = None, response_text: str = MOCK_RESPONSE
) -> LLMRouter:
    """Router serving the chat/synthesis model, the query router model, and the agent tool model."""
    chat_routed = RoutedLLM(
        adapter=chat_adapter or MockStreamingLLM(response_text=response_text),  # type: ignore[arg-type]
        provider="mock",
        model_id="gpt-4o-mini",
        default_params={"temperature": 0.2, "max_tokens": 2000},
        default_stream=True,
        capabilities={},
    )
    tool_routed = RoutedLLM(
        adapter=MockToolLLM(),  # type: ignore[arg-type]
        provider="mock",
        model_id=TOOL_MODEL_ID,
        default_params={},
        default_stream=False,
        capabilities={"tool_calling": True},
    )
    config = {
        "defaults": {"stream": True, "params": {"temperature": 0.2, "max_tokens": 2000}},
        "models": [],
    }
    router = LLMRouter(config)
    router._models["gpt-4o-mini"] = chat_routed
    router._models[TOOL_MODEL_ID] = tool_routed
    return router


@asynccontextmanager
async def app_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan that uses mock LLM router."""
    await init_db()
    app.state.llm_router = create_agentic_router()
    app.state.redis = await create_redis_app_client()

    yield

    await app.state.llm_router.close()
    await close_redis_client(app.state.redis)
    await shutdown_db()


@pytest.fixture(autouse=True)
def _celery_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run Celery tasks synchronously in tests; execute task logic on worker loop in a thread."""
    from src.celery_app import celery_app
    from src.services.chat.tasks import (
        _get_worker_loop,
        _initialize_worker_resources,
        _run_chat_pipeline,
        process_chat,
    )

    celery_app.conf.update(task_always_eager=True, task_eager_propagates=True)

    def _run_in_thread(request_id: str) -> None:
        _initialize_worker_resources()
        _get_worker_loop().run_until_complete(_run_chat_pipeline(request_id))

    def _patched_run(request_id: str) -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            _run_in_thread(request_id)
            return
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            pool.submit(_run_in_thread, request_id).result()

    monkeypatch.setattr(process_chat, "run", _patched_run)


@pytest.fixture(autouse=True)
def _patch_llm_router(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure get_router returns mock in integration tests (no real API calls)."""
    monkeypatch.setenv("AGENT_TOOL_MODEL", TOOL_MODEL_ID)
    mock_router = create_agentic_router()
    monkeypatch.setattr("src.services.llm_router.get_router", lambda *_args, **_kwargs: mock_router)
    monkeypatch.setattr("src.services.chat.tasks.get_router", lambda *_args, **_kwargs: mock_router)
    monkeypatch.setattr("src.services.chat.tasks._router", mock_router)


@pytest.fixture(autouse=True)
def _patch_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub out embedding calls (no TEI/OpenAI service available in these tests)."""
    monkeypatch.setattr(
        "src.services.retrieval.chat_rag.embed_chunks",
        lambda chunks: [[0.0] * 8 for _ in chunks],
    )


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
