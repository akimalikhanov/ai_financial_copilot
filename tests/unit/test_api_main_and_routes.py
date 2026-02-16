"""Tests for FastAPI app and chat routes."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

import src.main as main
from src.api.exceptions import _sse_event
from src.services.llm_adapters.base_adapter import LLMResponse, LLMStreamChunk
from src.services.llm_runtime.exceptions import LLMError


def test_create_app_registers_routes_and_handler():
    from src.api.exceptions import llm_error_handler

    app = main.create_app()

    routes = [r for r in app.router.routes if isinstance(r, APIRoute)]
    paths = {(r.path, tuple(sorted(r.methods or []))) for r in routes}

    assert ("/v1/chat", ("POST",)) in paths
    assert ("/v1/chat/stream", ("GET",)) in paths
    assert app.exception_handlers.get(LLMError) is llm_error_handler


def test_lifespan_sets_router_and_closes(monkeypatch: pytest.MonkeyPatch):
    router = DummyRouter(DummyLLM())
    monkeypatch.setattr(main, "get_router", lambda: router)

    mock_redis = MagicMock()
    mock_redis.aclose = AsyncMock()
    monkeypatch.setattr(main, "create_redis_client", AsyncMock(return_value=mock_redis))

    app = main.create_app()
    with TestClient(app):
        assert app.state.llm_router is router
    assert router.closed is True


def test_sse_event_escapes_non_ascii():
    payload = {"text": "café"}
    event = _sse_event("delta", payload)
    assert "event: delta" in event
    assert "\\u00e9" in event


class DummyLLM:
    def __init__(self, *, response_text: str = "ok") -> None:
        self.provider = "dummy"
        self.model_id = "dummy-model"
        self._response_text = response_text

    async def complete(self, *_, **__) -> LLMResponse:
        return LLMResponse(text=self._response_text)

    def stream(self, *_, **__) -> AsyncGenerator[LLMStreamChunk, None]:
        async def _gen() -> AsyncGenerator[LLMStreamChunk, None]:
            yield LLMStreamChunk(text="", is_final=True)

        return _gen()


class DummyRouter:
    def __init__(self, llm: DummyLLM) -> None:
        self._llm = llm
        self.closed = False

    def get(self, _: str) -> DummyLLM:
        return self._llm

    async def close(self) -> None:
        self.closed = True
