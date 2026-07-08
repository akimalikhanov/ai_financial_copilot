"""SMOKE ONLY — proves the agent loop starts, calls tools, and terminates.

Does NOT test termination-quality heuristics or round-count correctness beyond
the iteration cap; that is deferred to Stage 16c (sufficiency-evaluated termination).
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fakeredis import FakeAsyncRedis

from src.schemas.chat import ChatPipelineState
from src.schemas.query_router import DocumentScopeResult, RouterOutput
from src.schemas.retrieval import ChunkPromptPayload, RetrievedChunk
from src.services.chat import agent_loop as agent_loop_module
from src.services.chat.agent_loop import run_agent_loop
from src.services.llm_adapters.base_adapter import AssistantTurnResult, ToolCallRef
from src.services.llm_router import RoutedLLM


def _make_chunk_with_payload() -> tuple[RetrievedChunk, dict]:
    chunk_id = uuid4()
    document_id = uuid4()
    chunk = RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        score=1.0,
        chunk_index=0,
        page_start=1,
        page_end=1,
        heading_trail=[],
        source="vector",
    )
    payload = ChunkPromptPayload(
        chunk_id=chunk_id,
        document_id=document_id,
        document_name="Acme 10-K",
        page_numbers=(1,),
        heading_trail=(),
        prompt_text="[__REF__ | Acme 10-K | p.1]\nRevenue was $100.",
    )
    return chunk, {chunk_id: payload}


def _make_state(**scope_kwargs: Any) -> ChatPipelineState:
    router_output = RouterOutput(
        route="retrieval",
        entities=[],
        user_intent="test",
        reasoning="test",
        query_shape="extraction",
    )
    scope_result = DocumentScopeResult(
        doc_ids=None,
        source="all",
        per_entity_doc_ids={"Acme": [uuid4()]},
        entity_manifest=None,
        **scope_kwargs,
    )
    return ChatPipelineState(
        request_id=str(uuid4()),
        redis_app=FakeAsyncRedis(),
        session=AsyncMock(),
        conversation_id=uuid4(),
        user_query_raw="What was Acme's revenue?",
        context_messages=[],
        router_output=router_output,
        scope_result=scope_result,
    )


def _routed_llm(adapter: Any) -> RoutedLLM:
    return RoutedLLM(
        adapter=adapter,
        provider="mock",
        model_id="mock-tool-model",
        default_params={},
        default_stream=False,
        capabilities={"tool_calling": True},
    )


@pytest.fixture(autouse=True)
def _agent_config_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_LOOP_ENABLED", "true")
    monkeypatch.setenv("AGENT_MAX_ITERATIONS", "3")
    monkeypatch.setenv("AGENT_TOKEN_BUDGET", "1000000")
    monkeypatch.setenv("AGENT_MAX_CONCURRENT_SEARCHES", "1")
    monkeypatch.setenv("AGENT_MAX_CHUNKS_PER_ENTITY", "5")


@pytest.mark.asyncio
async def test_agent_loop_runs_search_then_finalizes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Loop issues a search_documents call, then report_findings, and terminates naturally."""
    state = _make_state()

    search_tc = ToolCallRef(
        id="call_1",
        name="search_documents",
        arguments=json.dumps({"entity": "Acme", "query": "revenue"}),
    )
    findings_tc = ToolCallRef(
        id="call_2",
        name="report_findings",
        arguments=json.dumps(
            {
                "metric_requested": "revenue",
                "findings": [{"entity": "Acme", "available": True, "value": 100}],
            }
        ),
    )

    adapter = AsyncMock()
    adapter.complete_with_tools = AsyncMock(
        side_effect=[
            AssistantTurnResult(text="", tool_calls=[search_tc]),
            AssistantTurnResult(text="", tool_calls=[findings_tc]),
        ]
    )
    llm = _routed_llm(adapter)

    found_chunk, payloads = _make_chunk_with_payload()
    monkeypatch.setattr(
        "src.services.chat.agent_loop._execute_search",
        AsyncMock(
            return_value=agent_loop_module._SearchResult(chunks=[found_chunk], payloads=payloads)
        ),
    )

    chunk_registry, agent_findings, meta = await run_agent_loop(
        state, llm, state.session, state.redis_app, state.request_id, reranker=None
    )

    assert meta.iterations == 2
    assert meta.convergence_reason == "natural"
    assert agent_findings is not None
    assert chunk_registry  # search chunk was admitted to the registry


@pytest.mark.asyncio
async def test_agent_loop_stops_at_iteration_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """An LLM mock that would loop forever is still bounded by max_iterations."""
    state = _make_state()

    def _always_search(*_args: Any, **_kwargs: Any) -> AssistantTurnResult:
        tc = ToolCallRef(
            id=str(uuid4()),
            name="search_documents",
            arguments=json.dumps({"entity": "Acme", "query": "revenue"}),
        )
        return AssistantTurnResult(text="", tool_calls=[tc])

    adapter = AsyncMock()
    adapter.complete_with_tools = AsyncMock(side_effect=_always_search)
    llm = _routed_llm(adapter)

    async def _fake_execute_search(*_args: Any, **_kwargs: Any):
        chunk, payloads = _make_chunk_with_payload()
        return agent_loop_module._SearchResult(chunks=[chunk], payloads=payloads)

    monkeypatch.setattr(
        "src.services.chat.agent_loop._execute_search",
        _fake_execute_search,
    )

    _chunk_registry, agent_findings, meta = await run_agent_loop(
        state, llm, state.session, state.redis_app, state.request_id, reranker=None
    )

    assert meta.iterations == 3  # AGENT_MAX_ITERATIONS
    assert meta.convergence_reason == "iteration_cap"
    assert agent_findings is None
