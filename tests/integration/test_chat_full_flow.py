"""
Integration test: full flow API → queue → worker → DB + SSE.

Requires: PostgreSQL (via PgBouncer) and Redis running (e.g. docker-compose up -d postgres pgbouncer redis).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from uuid import uuid4

import pytest

from src.workers.chat_worker import run_consume_loop
from tests.integration.conftest import MOCK_RESPONSE


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chat_full_flow_api_queue_worker_sse(async_client, integration_app) -> None:
    """
    Full flow: register → login → create conversation → enqueue chat → worker processes → SSE → DB updated.
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

    shutdown = asyncio.Event()
    worker_task = asyncio.create_task(
        run_consume_loop(
            integration_app.state.redis,
            integration_app.state.llm_router,
            shutdown,
            block_ms=300,
        )
    )
    try:
        # 1. Create conversation (authenticated)
        conv_resp = await async_client.post(
            "/v1/conversations",
            json={"title": "Integration test"},
            headers=headers,
        )
        assert conv_resp.status_code == 200
        conv_data = conv_resp.json()
        conversation_id = conv_data["conversation_id"]

        # 2. Enqueue chat (API → Redis queue, authenticated)
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

    finally:
        shutdown.set()
        worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task
