"""
SMOKE ONLY — proves the agentic chat pipeline still wires together end-to-end
and that its stage-count logging is self-consistent. Not a duplicate of
test_chat_full_flow.py's SSE assertions; this targets pipeline.stage log lines.

Requires: PostgreSQL (via PgBouncer) and Redis (redis-app, redis-broker) running.
"""

from __future__ import annotations

import json
import logging
from uuid import uuid4

import pytest

from src.utils.config import get_injection_scan_user_input_enabled


async def _run_chat_flow(async_client, headers: dict, conversation_id: str) -> None:
    enqueue_resp = await async_client.post(
        "/v1/chat",
        json={
            "conversation_id": str(conversation_id),
            "content": "What was Acme's revenue?",
            "client_msg_id": str(uuid4()),
            "client_request_id": str(uuid4()),
            "model": "gpt-4o-mini",
            "params": {},
        },
        headers=headers,
    )
    assert enqueue_resp.status_code == 200
    request_id = enqueue_resp.json()["request_id"]

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
                    if current_event == "usage" and data.get("persisted") is True:
                        return
                    current_event = None
                except json.JSONDecodeError:
                    pass


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agentic_mode_stage_count(async_client, caplog: pytest.LogCaptureFixture) -> None:
    """A retrieval-route request logs every stage it announced: six unconditional
    stages, agent_loop, and scan_user_input when the injection guardrail is on."""
    email = f"chatflow-agentic-{uuid4().hex}@test.com"
    reg = await async_client.post(
        "/v1/auth/register", json={"email": email, "password": "testpass123"}
    )
    assert reg.status_code == 200
    token = reg.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    conv_resp = await async_client.post(
        "/v1/conversations", json={"title": "Agentic smoke test"}, headers=headers
    )
    assert conv_resp.status_code == 200
    conversation_id = conv_resp.json()["conversation_id"]

    with caplog.at_level(logging.INFO, logger="src.services.chat.tasks"):
        await _run_chat_flow(async_client, headers, conversation_id)

    stage_logs = [r.message for r in caplog.records if "pipeline.stage" in r.message]
    assert any("agent_loop" in m for m in stage_logs), (
        f"agent_loop stage should be logged in agentic mode, got: {stage_logs}"
    )
    expected = 7 + int(get_injection_scan_user_input_enabled())
    assert len(stage_logs) == expected, f"Expected {expected} stages, got: {stage_logs}"
    assert all(f"/{expected}]" in m for m in stage_logs)
