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
from tests.integration.conftest import MOCK_RESPONSE, MockStreamingLLM, create_agentic_router


class _MockErrorLLM(MockStreamingLLM):
    """Mock LLM that routes normally, then raises during the synthesis stream."""

    def __init__(self, error_msg: str = "Simulated streaming error") -> None:
        super().__init__()
        self._error_msg = error_msg

    def stream(self, *_args: Any, **_kwargs: Any) -> AsyncGenerator[LLMStreamChunk, None]:
        async def _gen() -> AsyncGenerator[LLMStreamChunk, None]:
            raise RuntimeError(self._error_msg)
            yield  # unreachable, makes _gen an async generator

        return _gen()


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
    error_router = create_agentic_router(chat_adapter=_MockErrorLLM(error_msg))
    monkeypatch.setattr("src.services.llm_router.get_router", lambda *_a, **_k: error_router)
    monkeypatch.setattr("src.services.chat.tasks.get_router", lambda *_a, **_k: error_router)
    monkeypatch.setattr("src.services.chat.tasks._router", error_router)

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


_AURORA_ROUTER_JSON = json.dumps(
    {
        "route": "retrieval",
        "entities": [{"name": "Aurora", "entity_type": "company", "raw_span": "aurora"}],
        "user_intent": "single_lookup",
        "reasoning": "names aurora",
        "query_shape": "extraction",
    }
)


async def _chat(async_client, headers: dict, conversation_id: str, **body: Any) -> list:
    """POST /v1/chat, then read its SSE stream until the persisted usage event."""
    resp = await async_client.post(
        "/v1/chat",
        json={
            "conversation_id": conversation_id,
            "client_msg_id": str(uuid4()),
            "client_request_id": str(uuid4()),
            "model": "gpt-4o-mini",
            "params": {},
            **body,
        },
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    events: list[tuple[str, dict]] = []
    async with async_client.stream(
        "GET",
        "/v1/chat/stream",
        params={"request_id": resp.json()["request_id"]},
        timeout=15.0,
        headers=headers,
    ) as stream_response:
        current_event: str | None = None
        async for line in stream_response.aiter_lines():
            if line.startswith("event: "):
                current_event = line[7:].strip()
            elif line.startswith("data: ") and current_event:
                data = json.loads(line[6:])
                events.append((current_event, data))
                if current_event in ("error",) or (
                    current_event == "usage" and data.get("persisted") is True
                ):
                    break
    return events


@pytest.mark.integration
@pytest.mark.asyncio
async def test_clarification_card_reply_and_binding(
    async_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two Auroras: the question gets a card, a pick answers it, a follow-up uses the binding."""
    from sqlalchemy import select

    from src.db import get_session_factory
    from src.models.document import Document
    from src.models.user import User
    from src.services.router.company_name import normalize_company

    # The router names "aurora"; the disambiguator gets the same mock reply, can't parse it,
    # and falls back to the trigram candidates: both Auroras, so the card asks.
    router = create_agentic_router(MockStreamingLLM(router_json=_AURORA_ROUTER_JSON))
    monkeypatch.setattr("src.services.llm_router.get_router", lambda *_a, **_k: router)
    monkeypatch.setattr("src.services.chat.tasks.get_router", lambda *_a, **_k: router)
    monkeypatch.setattr("src.services.chat.tasks._router", router)

    email = f"clarify-{uuid4().hex}@test.com"
    reg = await async_client.post(
        "/v1/auth/register", json={"email": email, "password": "testpass123"}
    )
    headers = {"Authorization": f"Bearer {reg.json()['access_token']}"}
    async with get_session_factory()() as s:
        user_id = (await s.execute(select(User.id).where(User.email == email))).scalar_one()
        for company in ("Aurora Innovation, Inc.", "Aurora Mobile Limited"):
            s.add(
                Document(
                    user_id=user_id,
                    original_filename=f"{company}.pdf",
                    storage_key=f"uploads/{uuid4()}.pdf",
                    status="ready",
                    document_metadata={"company": company, "year": "2023"},
                    company_norm=normalize_company(company),
                )
            )
        await s.commit()
    conv = await async_client.post("/v1/conversations", json={"title": "c"}, headers=headers)
    conversation_id = str(conv.json()["conversation_id"])

    # 1. The question ends in a card, not an answer.
    events = await _chat(async_client, headers, conversation_id, content="aurora's revenue")
    [card] = [d for t, d in events if t == "scope_clarification"]
    assert card["outcome"] == "entities"
    [entity] = card["unresolved"]
    assert entity["outcome"] == "ambiguous"
    assert {c["company"] for c in entity["candidates"]} == {
        "Aurora Innovation, Inc.",
        "Aurora Mobile Limited",
    }

    # 2. A pick re-runs the question and answers it, without a second user message.
    reply = {
        "clarification_id": card["clarification_id"],
        "picks": [{"raw_span": "aurora", "company": "Aurora Innovation, Inc."}],
    }
    events = await _chat(
        async_client,
        headers,
        conversation_id,
        content="aurora's revenue",
        clarification_reply=reply,
    )
    assert not [d for t, d in events if t == "scope_clarification"]
    assert MOCK_RESPONSE in "".join(d.get("text", "") for t, d in events if t == "delta")
    msgs = (
        await async_client.get(f"/v1/conversations/{conversation_id}/messages", headers=headers)
    ).json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "assistant"]
    assert msgs[1]["metadata"]["kind"] == "clarification"
    assert msgs[1]["metadata"]["answered"] is True

    # 3. A later mention of "aurora" uses the binding: no second card.
    events = await _chat(async_client, headers, conversation_id, content="and aurora's debt?")
    assert not [d for t, d in events if t == "scope_clarification"]
    assert MOCK_RESPONSE in "".join(d.get("text", "") for t, d in events if t == "delta")
