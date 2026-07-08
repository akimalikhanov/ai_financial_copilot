"""
SMOKE ONLY — proves the agentic chat pipeline still wires together end-to-end
and documents a known quirk in its stage-count logging. Not a duplicate of
test_chat_full_flow.py's SSE assertions; this targets pipeline.stage log lines.

Classic (non-agent) mode is deprecated — AGENT_LOOP_ENABLED=true is the only
supported configuration, so this file only covers the agentic route.

Requires: PostgreSQL (via PgBouncer) and Redis (redis-app, redis-broker) running.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any
from uuid import uuid4

import pytest

from src.services.llm_adapters.base_adapter import (
    AssistantTurnResult,
    LLMResponse,
    LLMStreamChunk,
    ToolCallRef,
)
from src.services.llm_router import LLMRouter, RoutedLLM
from tests.integration.conftest import MOCK_RESPONSE

_TOOL_MODEL_ID = "mock-tool-model"

_RETRIEVAL_ROUTER_JSON = json.dumps(
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


class _MockRouterLLM:
    """Mock LLM answering the router's `.complete()` call and the final `.stream()` call."""

    def __init__(self, router_json: str, stream_text: str = MOCK_RESPONSE) -> None:
        self.provider = "mock"
        self.model_id = "mock-model"
        self._router_json = router_json
        self._stream_text = stream_text

    async def close(self) -> None:
        pass

    async def complete(self, *_args: Any, **_kwargs: Any) -> LLMResponse:
        return LLMResponse(text=self._router_json, stats=None)

    def stream(self, *_args: Any, **_kwargs: Any) -> AsyncGenerator[LLMStreamChunk, None]:
        async def _gen() -> AsyncGenerator[LLMStreamChunk, None]:
            yield LLMStreamChunk(text=self._stream_text, is_final=False)
            yield LLMStreamChunk(text="", is_final=True, stats=None)

        return _gen()


class _MockToolLLM:
    """Mock tool-calling LLM: finalizes immediately with report_findings (no search)."""

    def __init__(self) -> None:
        self.provider = "mock"
        self.model_id = _TOOL_MODEL_ID

    async def close(self) -> None:
        pass

    async def complete_with_tools(self, *_args: Any, **_kwargs: Any) -> AssistantTurnResult:
        return AssistantTurnResult(text="", tool_calls=[_REPORT_FINDINGS_TC])


def _create_agentic_router() -> LLMRouter:
    """Router serving the chat model, query router model, and agent tool model."""
    chat_llm = _MockRouterLLM(_RETRIEVAL_ROUTER_JSON)
    chat_routed = RoutedLLM(
        adapter=chat_llm,  # type: ignore[arg-type]
        provider="mock",
        model_id="gpt-4o-mini",
        default_params={"temperature": 0.2, "max_tokens": 2000},
        default_stream=True,
        capabilities={},
    )
    tool_routed = RoutedLLM(
        adapter=_MockToolLLM(),  # type: ignore[arg-type]
        provider="mock",
        model_id=_TOOL_MODEL_ID,
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
    router._models[_TOOL_MODEL_ID] = tool_routed
    return router


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
async def test_agentic_mode_stage_count(
    async_client, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Agentic mode logs exactly 8 stages, matching tasks.py's hardcoded stage_total=8:
    agent_loop replaces transform_query/build_rag_context one-for-one rather than
    adding a 9th stage. Documents current, real behavior — not a design endorsement.
    """
    monkeypatch.setenv("AGENT_LOOP_ENABLED", "true")
    monkeypatch.setenv("AGENT_TOOL_MODEL", _TOOL_MODEL_ID)

    agentic_router = _create_agentic_router()
    monkeypatch.setattr("src.services.llm_router.get_router", lambda *_a, **_k: agentic_router)
    monkeypatch.setattr("src.services.chat.tasks.get_router", lambda *_a, **_k: agentic_router)
    monkeypatch.setattr("src.services.chat.tasks._router", agentic_router)

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
    assert not any("transform_query" in m for m in stage_logs), (
        f"transform_query is classic-mode only, must not run in agentic mode, got: {stage_logs}"
    )
    assert len(stage_logs) == 8, f"Expected exactly 8 stages in agentic mode, got: {stage_logs}"
    assert all("/8]" in m for m in stage_logs)
