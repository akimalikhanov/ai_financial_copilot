"""
Integration test: full flow API → Celery task → DB + SSE.

Requires: PostgreSQL (via PgBouncer) and Redis (redis-app, redis-broker) running
(e.g. docker-compose up -d postgres pgbouncer redis-app redis-broker).
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any
from uuid import uuid4

import pytest

from src.redis_client import get_chat_tail
from src.services.llm_adapters.base_adapter import LLMStreamChunk
from src.services.llm_router import LLMRouter, RoutedLLM
from tests.integration.conftest import MOCK_RESPONSE


class _MockErrorLLM:
    """Mock LLM that raises during streaming."""

    def __init__(self, error_msg: str = "Simulated streaming error") -> None:
        self.provider = "mock"
        self.model_id = "mock-model"
        self._error_msg = error_msg

    async def close(self) -> None:
        pass

    def stream(self, *_args: Any, **_kwargs: Any) -> AsyncGenerator[LLMStreamChunk, None]:
        async def _gen() -> AsyncGenerator[LLMStreamChunk, None]:
            raise RuntimeError(self._error_msg)
            yield  # unreachable, makes _gen an async generator

        return _gen()


def _create_error_router(error_msg: str = "Simulated streaming error") -> LLMRouter:
    mock_llm = _MockErrorLLM(error_msg=error_msg)
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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chat_full_flow_api_queue_worker_sse(async_client) -> None:
    """
    Full flow: register → login → create conversation → POST chat (Celery eager) → SSE → DB updated.
    """
    # 0. Register and get token (unique email so reruns don't get 409)
    email = f"chatflow-{uuid4().hex}@test.com"
    reg = await async_client.post(
        "/v1/auth/register",
        json={"email": email, "password": "testpass123"},
    )
    assert reg.status_code == 200
    token = reg.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Create conversation (authenticated)
    conv_resp = await async_client.post(
        "/v1/conversations",
        json={"title": "Integration test"},
        headers=headers,
    )
    assert conv_resp.status_code == 200
    conv_data = conv_resp.json()
    conversation_id = conv_data["conversation_id"]

    # 2. POST chat (triggers process_chat.delay() in eager mode)
    enqueue_resp = await async_client.post(
        "/v1/chat",
        json={
            "conversation_id": str(conversation_id),
            "content": "Hello",
            "client_msg_id": str(uuid4()),
            "client_request_id": str(uuid4()),
            "model": "gpt-4o-mini",
            "params": {},
        },
        headers=headers,
    )
    assert enqueue_resp.status_code == 200
    enqueue_data = enqueue_resp.json()
    request_id = enqueue_data["request_id"]

    # 3. Connect to SSE stream and collect events until usage
    events: list[tuple[str, dict]] = []
    timeout_seconds = 15.0

    async with async_client.stream(
        "GET",
        "/v1/chat/stream",
        params={"request_id": str(request_id)},
        timeout=timeout_seconds,
        headers=headers,
    ) as stream_response:
        assert stream_response.status_code == 200
        current_event: str | None = None
        async for line in stream_response.aiter_lines():
            if line.startswith("event: "):
                current_event = line[7:].strip()
            elif line.startswith("data: ") and current_event:
                try:
                    data = json.loads(line[6:])
                    events.append((current_event, data))
                    if current_event == "usage" and data.get("persisted") is True:
                        break
                    current_event = None
                except json.JSONDecodeError:
                    pass

    # 4. Assert we got expected events
    assert len(events) >= 1, f"Expected at least one event, got: {events}"

    delta_events = [(t, d) for t, d in events if t == "delta"]
    usage_events = [(t, d) for t, d in events if t == "usage" and d.get("persisted")]

    assert len(delta_events) >= 1, f"Expected delta events, got: {events}"
    assert len(usage_events) >= 1, f"Expected usage with persisted, got: {events}"

    combined_text = "".join(d.get("text", "") for _, d in delta_events)
    assert MOCK_RESPONSE in combined_text, (
        f"Expected mock response, got real LLM output: {combined_text!r}"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chat_tail_cache_populated_after_flow(async_client, integration_app) -> None:
    """Verify chat tail cache is populated after full API→worker flow."""
    email = f"chatflow-cache-{uuid4().hex}@test.com"
    reg = await async_client.post(
        "/v1/auth/register",
        json={"email": email, "password": "testpass123"},
    )
    assert reg.status_code == 200
    token = reg.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    conv_resp = await async_client.post(
        "/v1/conversations",
        json={"title": "Cache test"},
        headers=headers,
    )
    assert conv_resp.status_code == 200
    conversation_id = conv_resp.json()["conversation_id"]

    enqueue_resp = await async_client.post(
        "/v1/chat",
        json={
            "conversation_id": str(conversation_id),
            "content": "Hello",
            "client_msg_id": str(uuid4()),
            "client_request_id": str(uuid4()),
            "model": "gpt-4o-mini",
            "params": {},
        },
        headers=headers,
    )
    assert enqueue_resp.status_code == 200
    request_id = enqueue_resp.json()["request_id"]

    events: list[tuple[str, dict]] = []
    async with async_client.stream(
        "GET",
        "/v1/chat/stream",
        params={"request_id": str(request_id)},
        timeout=15.0,
        headers=headers,
    ) as stream_response:
        current_event: str | None = None
        async for line in stream_response.aiter_lines():
            if line.startswith("event: "):
                current_event = line[7:].strip()
            elif line.startswith("data: ") and current_event:
                try:
                    data = json.loads(line[6:])
                    events.append((current_event, data))
                    if current_event == "usage" and data.get("persisted") is True:
                        break
                    current_event = None
                except json.JSONDecodeError:
                    pass

    usage_events = [(t, d) for t, d in events if t == "usage" and d.get("persisted")]
    assert len(usage_events) >= 1, "Flow should complete with persisted usage"

    redis = integration_app.state.redis
    cached = await get_chat_tail(redis, str(conversation_id))
    assert cached is not None, "Chat tail cache should be populated after flow"
    msgs, latest_seq = cached
    assert len(msgs) == 2, "Cache should have user + assistant messages"
    assert latest_seq == 2
    assert msgs[0].get("role") == "user" and msgs[0].get("content") == "Hello"
    assert msgs[1].get("role") == "assistant" and MOCK_RESPONSE in (msgs[1].get("content") or "")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chat_error_propagation_sse(async_client, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify SSE emits structured error event when LLM raises during streaming."""
    error_msg = "Simulated streaming error"
    error_router = _create_error_router(error_msg)
    monkeypatch.setattr("src.services.llm_router.get_router", lambda *_a, **_k: error_router)
    monkeypatch.setattr("src.workers.chat_worker.get_router", lambda *_a, **_k: error_router)

    email = f"chatflow-err-{uuid4().hex}@test.com"
    reg = await async_client.post(
        "/v1/auth/register",
        json={"email": email, "password": "testpass123"},
    )
    assert reg.status_code == 200
    token = reg.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    conv_resp = await async_client.post(
        "/v1/conversations",
        json={"title": "Error test"},
        headers=headers,
    )
    assert conv_resp.status_code == 200
    conversation_id = conv_resp.json()["conversation_id"]

    enqueue_resp = await async_client.post(
        "/v1/chat",
        json={
            "conversation_id": str(conversation_id),
            "content": "Hello",
            "client_msg_id": str(uuid4()),
            "client_request_id": str(uuid4()),
            "model": "gpt-4o-mini",
            "params": {},
        },
        headers=headers,
    )
    assert enqueue_resp.status_code == 200
    request_id = enqueue_resp.json()["request_id"]

    events: list[tuple[str, dict]] = []
    async with async_client.stream(
        "GET",
        "/v1/chat/stream",
        params={"request_id": str(request_id)},
        timeout=15.0,
        headers=headers,
    ) as stream_response:
        assert stream_response.status_code == 200
        current_event: str | None = None
        async for line in stream_response.aiter_lines():
            if line.startswith("event: "):
                current_event = line[7:].strip()
            elif line.startswith("data: ") and current_event:
                try:
                    data = json.loads(line[6:])
                    events.append((current_event, data))
                    if current_event == "error":
                        break
                    current_event = None
                except json.JSONDecodeError:
                    pass

    error_events = [(t, d) for t, d in events if t == "error"]
    assert len(error_events) >= 1, f"Expected error event, got: {events}"
    _, err_data = error_events[0]
    assert "message" in err_data
    assert "error_type" in err_data
    assert err_data["message"] == error_msg
    assert err_data["error_type"] == "RuntimeError"
