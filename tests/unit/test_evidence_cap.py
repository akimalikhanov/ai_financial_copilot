"""P2 regression test (docs/stages/agentic_state_refactor_v2.md step 6).

Mid-loop, the model must see at most `max_chunks_per_entity` rendered excerpts per
search — that cap moved from a post-loop registry mutation to a render-time
projection at transcript entry. `EvidenceLedger` must still hold the *full* admitted
set regardless (provenance intact for P1-5's any-lookup-top-N accounting).
"""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import AsyncIterator
from typing import Any, cast
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fakeredis import FakeAsyncRedis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.schemas.chat import ChatPipelineState
from src.schemas.query_router import DocumentScopeResult, RouterOutput
from src.schemas.retrieval import ChunkPromptPayload, RetrievedChunk
from src.services.chat.agent.loop import _SearchResult, run_loop
from src.services.llm_adapters.base_adapter import AssistantTurnResult, Role, ToolCallRef
from src.services.llm_router import RoutedLLM

_LABEL_RE = re.compile(r'id="(S\d+)"')


def _fake_session_factory() -> async_sessionmaker[AsyncSession]:
    def _factory() -> Any:
        @contextlib.asynccontextmanager
        async def _cm() -> AsyncIterator[Any]:
            yield AsyncMock()

        return _cm()

    return cast("async_sessionmaker[AsyncSession]", _factory)


def _make_chunks_with_payloads(n: int) -> tuple[list[RetrievedChunk], dict]:
    chunks: list[RetrievedChunk] = []
    payloads: dict = {}
    for i in range(n):
        chunk_id = uuid4()
        document_id = uuid4()
        chunk = RetrievedChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            score=float(n - i),  # descending — already in reranker order
            chunk_index=i,
            page_start=1,
            page_end=1,
            heading_trail=[],
            source="vector",
        )
        chunks.append(chunk)
        payloads[chunk_id] = ChunkPromptPayload(
            chunk_id=chunk_id,
            document_id=document_id,
            document_name="Acme 10-K",
            page_numbers=(1,),
            heading_trail=(),
            prompt_text=f"[__REF__ | Acme 10-K | p.1]\nExcerpt {i}.",
        )
    return chunks, payloads


def _make_state() -> ChatPipelineState:
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
    monkeypatch.setenv("AGENT_MAX_CHUNKS_PER_ENTITY", "2")


@pytest.mark.asyncio
async def test_mid_loop_tool_message_capped_but_ledger_holds_full_set() -> None:
    state = _make_state()
    chunks, payloads = _make_chunks_with_payloads(5)  # cap (env above) is 2

    search_tc = ToolCallRef(
        id="call_1",
        name="search_documents",
        arguments=json.dumps({"entity": "Acme", "query": "revenue"}),
    )
    # Cite an admitted chunk so the finding grounds and the coverage gate (which now
    # requires each searched entity to be reported) accepts the finalizer.
    findings_tc = ToolCallRef(
        id="call_2",
        name="report_findings",
        arguments=json.dumps(
            {
                "metric_requested": "revenue",
                "findings": [
                    {
                        "entity": "Acme",
                        "available": True,
                        "value": 100,
                        "source_chunks": [str(chunks[0].chunk_id)],
                    }
                ],
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

    async def _fake_execute_search(*_args: Any, **_kwargs: Any) -> _SearchResult:
        return _SearchResult(entity="Acme", chunks=chunks, payloads=payloads)

    chunk_registry, _findings, _meta = await run_loop(
        state,
        llm,
        state.session,
        state.redis_app,
        state.request_id,
        reranker=None,
        session_factory=_fake_session_factory(),
        execute_search=_fake_execute_search,
    )

    # The ledger (returned registry) admitted the full result — provenance intact.
    assert len(chunk_registry) == 5

    # But the second LLM call (after the search turn) must only ever have seen the
    # capped render in its tool message, not all 5 excerpts.
    assert len(adapter.complete_with_tools.call_args_list) == 2
    second_call_messages = adapter.complete_with_tools.call_args_list[1].kwargs["messages"]
    tool_msg = next(m for m in second_call_messages if m.role == Role.tool)
    assert len(_LABEL_RE.findall(tool_msg.content or "")) == 2
