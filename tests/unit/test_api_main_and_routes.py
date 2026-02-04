from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

import src.main as main
from src.api.exceptions import _sse_event
from src.services.llm_adapters.base_adapter import (
    LLMResponse,
    LLMResponseStats,
    LLMStreamChunk,
)
from src.services.llm_runtime.exceptions import LLMError, LLMRateLimitError


class DummyLLM:
    def __init__(
        self,
        *,
        response_text: str = "ok",
        stream_chunks: list[LLMStreamChunk] | None = None,
        stream_error: Exception | None = None,
        complete_error: Exception | None = None,
    ) -> None:
        self.provider = "dummy"
        self.model_id = "dummy-model"
        self._response_text = response_text
        self._stream_chunks = stream_chunks or []
        self._stream_error = stream_error
        self._complete_error = complete_error

    async def complete(self, *_, **__) -> LLMResponse:
        if self._complete_error:
            raise self._complete_error
        return LLMResponse(text=self._response_text)

    def stream(self, *_, **__) -> AsyncGenerator[LLMStreamChunk, None]:
        async def _gen() -> AsyncGenerator[LLMStreamChunk, None]:
            for chunk in self._stream_chunks:
                yield chunk
            if self._stream_error:
                raise self._stream_error

        return _gen()


class DummyRouter:
    def __init__(self, llm: DummyLLM, *, raise_on_get: Exception | None = None) -> None:
        self._llm = llm
        self._raise_on_get = raise_on_get
        self.closed = False

    def get(self, _: str) -> DummyLLM:
        if self._raise_on_get:
            raise self._raise_on_get
        return self._llm

    async def close(self) -> None:
        self.closed = True


def _chat_payload(model: str = "dummy-model") -> dict[str, object]:
    return {
        "messages": [{"role": "user", "content": "hi"}],
        "model": model,
        "temperature": 0.2,
        "max_tokens": 16,
        "extra_params": {},
    }


def _collect_sse_events(raw: str) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    for block in raw.strip().split("\n\n"):
        if not block:
            continue
        lines = block.splitlines()
        event = next(line for line in lines if line.startswith("event: ")).split("event: ", 1)[1]
        data = next(line for line in lines if line.startswith("data: ")).split("data: ", 1)[1]
        events.append((event, json.loads(data)))
    return events


def test_create_app_registers_routes_and_handler():
    from src.api.exceptions import llm_error_handler

    app = main.create_app()

    routes = [route for route in app.router.routes if isinstance(route, APIRoute)]
    paths = {(route.path, tuple(sorted(route.methods or []))) for route in routes}

    assert ("/v1/chat", ("POST",)) in paths
    assert ("/v1/chat/stream", ("POST",)) in paths
    assert app.exception_handlers.get(LLMError) is llm_error_handler


def test_lifespan_sets_router_and_closes(monkeypatch: pytest.MonkeyPatch):
    router = DummyRouter(DummyLLM())
    monkeypatch.setattr(main, "get_router", lambda: router)

    app = main.create_app()
    with TestClient(app):
        assert app.state.llm_router is router
    assert router.closed is True


def test_chat_endpoint_returns_response(monkeypatch: pytest.MonkeyPatch):
    router = DummyRouter(DummyLLM(response_text="hello"))
    monkeypatch.setattr(main, "get_router", lambda: router)

    app = main.create_app()
    with TestClient(app) as client:
        response = client.post("/v1/chat", json=_chat_payload())
    assert response.status_code == 200
    assert response.json() == {"text": "hello"}


def test_chat_endpoint_maps_llm_error_to_json(monkeypatch: pytest.MonkeyPatch):
    error = LLMRateLimitError("rate limit", status_code=429, error_code="rate_limit")
    router = DummyRouter(DummyLLM(complete_error=error))
    monkeypatch.setattr(main, "get_router", lambda: router)

    app = main.create_app()
    with TestClient(app) as client:
        response = client.post("/v1/chat", json=_chat_payload())

    assert response.status_code == 429
    payload = response.json()
    assert payload["error_type"] == "LLMRateLimitError"
    assert payload["message"] == "rate limit"
    assert payload["status_code"] == 429
    assert payload["error_code"] == "rate_limit"
    assert payload["is_retryable"] is True


def test_chat_stream_formats_sse_events_and_headers(monkeypatch: pytest.MonkeyPatch):
    stats = LLMResponseStats(input_tokens=1, output_tokens=2, total_tokens=3)
    chunks = [
        LLMStreamChunk(text="hi", is_final=False),
        LLMStreamChunk(text="done", is_final=True, stats=stats),
    ]
    router = DummyRouter(DummyLLM(stream_chunks=chunks))
    monkeypatch.setattr(main, "get_router", lambda: router)

    app = main.create_app()
    with TestClient(app) as client:
        response = client.post("/v1/chat/stream", json=_chat_payload())

    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["connection"] == "keep-alive"
    assert response.headers["x-accel-buffering"] == "no"

    events = _collect_sse_events(response.text)
    assert events[0][0] == "delta"
    assert events[0][1]["text"] == "hi"
    assert events[1][0] == "usage"
    assert events[1][1]["text"] == "done"
    assert events[1][1]["stats"]["total_tokens"] == 3


def test_chat_stream_emits_error_event_on_llm_error(monkeypatch: pytest.MonkeyPatch):
    chunks = [LLMStreamChunk(text="partial", is_final=False)]
    stream_error = LLMRateLimitError("rate limit", status_code=429)
    router = DummyRouter(DummyLLM(stream_chunks=chunks, stream_error=stream_error))
    monkeypatch.setattr(main, "get_router", lambda: router)

    app = main.create_app()
    with TestClient(app) as client:
        response = client.post("/v1/chat/stream", json=_chat_payload())

    events = _collect_sse_events(response.text)
    assert events[0][0] == "delta"
    assert events[1][0] == "error"
    assert events[1][1]["error_type"] == "LLMRateLimitError"
    assert events[1][1]["status_code"] == 429


def test_llm_error_handler_stream_route_returns_sse(monkeypatch: pytest.MonkeyPatch):
    router = DummyRouter(
        DummyLLM(),
        raise_on_get=LLMRateLimitError("rate limit", status_code=429),
    )
    monkeypatch.setattr(main, "get_router", lambda: router)

    app = main.create_app()
    with TestClient(app) as client:
        response = client.post("/v1/chat/stream", json=_chat_payload())

    events = _collect_sse_events(response.text)
    assert events[0][0] == "error"
    assert events[0][1]["error_type"] == "LLMRateLimitError"


def test_chat_stream_emits_internal_error_on_unexpected_exception(monkeypatch: pytest.MonkeyPatch):
    """Test catch-all Exception handler in stream emits InternalServerError."""

    class UnexpectedError(Exception):
        pass

    chunks = [LLMStreamChunk(text="partial", is_final=False)]
    router = DummyRouter(DummyLLM(stream_chunks=chunks, stream_error=UnexpectedError("boom")))
    monkeypatch.setattr(main, "get_router", lambda: router)

    app = main.create_app()
    with TestClient(app) as client:
        response = client.post("/v1/chat/stream", json=_chat_payload())

    events = _collect_sse_events(response.text)
    assert events[0][0] == "delta"
    assert events[1][0] == "error"
    assert events[1][1]["error_type"] == "InternalServerError"
    assert events[1][1]["message"] == "Internal server error"


def test_sse_event_escapes_non_ascii():
    payload = {"text": "café"}
    event = _sse_event("delta", payload)
    assert "event: delta" in event
    assert "\\u00e9" in event
