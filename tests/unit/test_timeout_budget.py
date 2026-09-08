"""Phase 3: every outbound call is bounded, and timing out degrades instead of failing.

The pathologies covered here are the ones that are invisible when everything is healthy:
a config key that is loaded but never applied, and a fan-out that sits outside the
per-turn timeout so nothing bounds it but the Celery hard limit.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fakeredis import FakeAsyncRedis

from src.schemas.chat import ChatPipelineState
from src.schemas.query_router import DocumentScopeResult, RouterOutput
from src.services.llm_adapters.base_adapter import AssistantTurnResult, ToolCallRef
from src.utils.config import (
    get_llm_connect_timeout_seconds,
    get_llm_max_retries,
    get_llm_timeout_seconds,
)

# --- 3.1 adapters carry the configured budget, not the SDK default ---


def test_openai_adapter_applies_configured_timeout_and_retries() -> None:
    """Unset, openai-python uses a 600s read timeout and 3 attempts."""
    import httpx

    from src.services.llm_adapters.openai_adapter import OpenAIAdapter

    client = OpenAIAdapter(default_model="m", api_key="k")._client

    timeout = client.timeout
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.read == get_llm_timeout_seconds()
    assert timeout.connect == get_llm_connect_timeout_seconds()
    assert client.max_retries == get_llm_max_retries()


def test_gemini_adapter_applies_configured_timeout_in_milliseconds() -> None:
    """google-genai takes milliseconds and counts the original request in `attempts`."""
    from src.services.llm_adapters.gemini_adapter import GeminiAdapter

    opts = GeminiAdapter(default_model="m", api_key="k")._client._api_client._http_options

    assert opts.timeout == int(get_llm_timeout_seconds() * 1000)
    assert opts.retry_options is not None
    assert opts.retry_options.attempts == get_llm_max_retries() + 1


# --- 3.2 the dead QUERY_TRANSFORMER_TIMEOUT is now enforced ---


class _HangingLLM:
    """An LLM that never answers — the degraded-upstream case the timeout exists for."""

    provider = "openai"
    model_id = "m"

    async def complete(self, **_kwargs: Any) -> Any:
        await asyncio.sleep(30)
        raise AssertionError("should have timed out")


class _HangingRouter:
    def get(self, _model_id: str) -> Any:
        return _HangingLLM()


@pytest.mark.asyncio
async def test_slow_rewrite_falls_back_to_the_raw_query(monkeypatch: pytest.MonkeyPatch) -> None:
    """A rewrite is an optimisation: exceeding its budget must degrade to the un-rewritten
    query, not fail the request. Before this the configured value was never applied and the
    effective timeout was the SDK's 600s."""
    monkeypatch.setenv("QUERY_TRANSFORMER_TIMEOUT", "0.05")

    from src.services.retrieval import query_transformer as qt

    started = asyncio.get_running_loop().time()
    result, _stats = await qt.rewrite_query(
        "what was revenue in 2023",
        llm_router=_HangingRouter(),  # type: ignore[arg-type]
    )
    elapsed = asyncio.get_running_loop().time() - started

    assert result.fallback is True
    assert result.semantic_query == "what was revenue in 2023"
    assert elapsed < 5.0, f"rewrite was not bounded by its timeout (took {elapsed:.1f}s)"


@pytest.mark.asyncio
async def test_rewrite_timeout_is_read_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raising the configured budget must actually lengthen the wait — proof the value is
    applied at the use site rather than merely loaded into cfg."""
    from src.services.retrieval import query_transformer as qt

    monkeypatch.setenv("QUERY_TRANSFORMER_TIMEOUT", "0.05")
    t0 = asyncio.get_running_loop().time()
    await qt.rewrite_query("q", llm_router=_HangingRouter())  # type: ignore[arg-type]
    short = asyncio.get_running_loop().time() - t0

    monkeypatch.setenv("QUERY_TRANSFORMER_TIMEOUT", "0.6")
    t0 = asyncio.get_running_loop().time()
    await qt.rewrite_query("q", llm_router=_HangingRouter())  # type: ignore[arg-type]
    longer = asyncio.get_running_loop().time() - t0

    assert longer > short


# --- 3.3 the search fan-out is bounded, and bounding it stays fail-open ---


def _fake_session_factory() -> Any:
    def _factory() -> Any:
        @contextlib.asynccontextmanager
        async def _cm() -> AsyncIterator[Any]:
            yield AsyncMock()

        return _cm()

    return _factory


def _make_state() -> ChatPipelineState:
    return ChatPipelineState(
        request_id=str(uuid4()),
        redis_app=FakeAsyncRedis(),
        session=AsyncMock(),
        conversation_id=uuid4(),
        user_query_raw="What was Acme's revenue?",
        context_messages=[],
        router_output=RouterOutput(
            route="retrieval",
            entities=[],
            user_intent="test",
            reasoning="test",
            query_shape="extraction",
        ),
        scope_result=DocumentScopeResult(
            doc_ids=None,
            source="all",
            per_entity_doc_ids={"Acme": [uuid4()]},
            entity_manifest=None,
        ),
    )


@pytest.mark.asyncio
async def test_hung_search_does_not_run_past_the_turn_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fan-out sits outside the per-turn wait_for, so before this only the Celery hard
    limit bounded it — one hung backend held a worker slot for 20 minutes."""
    monkeypatch.setenv("AGENT_MAX_ITERATIONS", "1")
    monkeypatch.setenv("AGENT_TOKEN_BUDGET", "1000000")
    monkeypatch.setenv("AGENT_MAX_CONCURRENT_SEARCHES", "2")
    monkeypatch.setenv("AGENT_TURN_TIMEOUT_SECONDS", "0.2")

    from src.services.chat.agent.loop import run_loop
    from src.services.llm_router import RoutedLLM

    state = _make_state()

    adapter = AsyncMock()
    adapter.complete_with_tools = AsyncMock(
        return_value=AssistantTurnResult(
            text="",
            tool_calls=[
                ToolCallRef(
                    id="call_1",
                    name="search_documents",
                    arguments=json.dumps({"entity": "Acme", "query": "revenue"}),
                )
            ],
        )
    )
    llm = RoutedLLM(
        adapter=adapter,
        provider="mock",
        model_id="mock-tool-model",
        default_params={},
        default_stream=False,
        capabilities={"tool_calling": True},
    )

    async def _hanging_search(*_args: Any, **_kwargs: Any) -> Any:
        await asyncio.sleep(30)
        raise AssertionError("should have timed out")

    started = asyncio.get_running_loop().time()
    _evidence, _findings, meta = await run_loop(
        state,
        llm,
        state.session,
        state.redis_app,
        state.request_id,
        reranker=None,
        session_factory=_fake_session_factory(),
        execute_search=_hanging_search,
    )
    elapsed = asyncio.get_running_loop().time() - started

    assert elapsed < 10.0, f"fan-out was not bounded (took {elapsed:.1f}s)"
    # Fail-open, not fail-closed: the run completes and reports, rather than raising.
    assert meta.iterations >= 1
