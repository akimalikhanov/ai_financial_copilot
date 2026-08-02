"""SMOKE ONLY — proves the agent loop starts, calls tools, and terminates.

Does NOT test termination-quality heuristics or round-count correctness beyond
the iteration cap; that is deferred to Stage 16c (sufficiency-evaluated termination).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from typing import Any, cast
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fakeredis import FakeAsyncRedis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.schemas.agent_findings import AnalyticalFindings, Observation
from src.schemas.chat import ChatPipelineState
from src.schemas.query_router import DocumentScopeResult, RouterOutput
from src.schemas.retrieval import ChunkPromptPayload, RetrievedChunk
from src.services.chat.agent import gates as gates_module
from src.services.chat.agent.loop import _SearchResult, run_loop
from src.services.llm_adapters.base_adapter import AssistantTurnResult, ToolCallRef
from src.services.llm_router import RoutedLLM


def _fake_session_factory() -> async_sessionmaker[AsyncSession]:
    """A callable yielding a fresh AsyncMock session per `async with` (mirrors
    `async_sessionmaker`). Each call returns a distinct session object."""

    def _factory() -> Any:
        @contextlib.asynccontextmanager
        async def _cm() -> AsyncIterator[Any]:
            yield AsyncMock()

        return _cm()

    return cast("async_sessionmaker[AsyncSession]", _factory)


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
async def test_agent_loop_runs_search_then_finalizes() -> None:
    """Loop issues a search_documents call, then report_findings, and terminates naturally."""
    state = _make_state()
    found_chunk, payloads = _make_chunk_with_payload()

    search_tc = ToolCallRef(
        id="call_1",
        name="search_documents",
        arguments=json.dumps({"entity": "Acme", "query": "revenue"}),
    )
    # The finding must cite the admitted chunk so it grounds (C6) and lands a ledger key —
    # the coverage gate now requires every searched entity to be *reported*, not just searched.
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
                        "source_chunks": [str(found_chunk.chunk_id)],
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
        return _SearchResult(entity="Acme", chunks=[found_chunk], payloads=payloads)

    chunk_registry, agent_findings, meta = await run_loop(
        state,
        llm,
        state.session,
        state.redis_app,
        state.request_id,
        reranker=None,
        session_factory=_fake_session_factory(),
        execute_search=_fake_execute_search,
    )

    assert meta.iterations == 2
    assert meta.convergence_reason == "natural"
    assert agent_findings is not None
    assert chunk_registry  # search chunk was admitted to the registry


@pytest.mark.asyncio
async def test_agent_loop_stops_at_iteration_cap() -> None:
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

    async def _fake_execute_search(*_args: Any, **_kwargs: Any) -> _SearchResult:
        chunk, payloads = _make_chunk_with_payload()
        return _SearchResult(entity="Acme", chunks=[chunk], payloads=payloads)

    _chunk_registry, agent_findings, meta = await run_loop(
        state,
        llm,
        state.session,
        state.redis_app,
        state.request_id,
        reranker=None,
        session_factory=_fake_session_factory(),
        execute_search=_fake_execute_search,
    )

    assert meta.iterations == 3  # AGENT_MAX_ITERATIONS
    assert meta.convergence_reason == "iteration_cap"
    assert agent_findings is None


@pytest.mark.asyncio
async def test_concurrent_searches_each_open_a_distinct_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P0-1: fanned-out searches must not share the loop's session — each opens its own.

    Fire N searches in one turn and assert N distinct sessions were opened from the
    factory (and that none of them is the loop's own serial `session`).
    """
    monkeypatch.setenv("AGENT_MAX_CONCURRENT_SEARCHES", "3")
    state = _make_state()
    # One shared chunk across all searches (they dedup on chunk_id); the finding cites it so
    # the coverage gate accepts the finalizer instead of forcing an unscripted extra turn.
    shared_chunk, shared_payloads = _make_chunk_with_payload()

    n = 3
    search_tcs = [
        ToolCallRef(
            id=f"call_{i}",
            name="search_documents",
            arguments=json.dumps({"entity": "Acme", "query": f"revenue {i}"}),
        )
        for i in range(n)
    ]
    findings_tc = ToolCallRef(
        id="call_fin",
        name="report_findings",
        arguments=json.dumps(
            {
                "metric_requested": "revenue",
                "findings": [
                    {
                        "entity": "Acme",
                        "available": True,
                        "value": 100,
                        "source_chunks": [str(shared_chunk.chunk_id)],
                    }
                ],
            }
        ),
    )
    adapter = AsyncMock()
    adapter.complete_with_tools = AsyncMock(
        side_effect=[
            AssistantTurnResult(text="", tool_calls=search_tcs),
            AssistantTurnResult(text="", tool_calls=[findings_tc]),
        ]
    )
    llm = _routed_llm(adapter)

    opened: list[Any] = []

    def _recording_factory() -> Any:
        @contextlib.asynccontextmanager
        async def _cm() -> AsyncIterator[Any]:
            sess = AsyncMock()
            opened.append(sess)
            yield sess

        return _cm()

    seen: list[Any] = []

    async def _fake_execute_search(_tc: Any, _state: Any, session: Any, *_a: Any, **_k: Any):
        seen.append(session)
        # Yield to the event loop so overlapping searches can't be papered over.
        await asyncio.sleep(0)
        return _SearchResult(entity="Acme", chunks=[shared_chunk], payloads=shared_payloads)

    await run_loop(
        state,
        llm,
        state.session,
        state.redis_app,
        state.request_id,
        reranker=None,
        session_factory=cast("async_sessionmaker[AsyncSession]", _recording_factory),
        execute_search=_fake_execute_search,
    )

    assert len(opened) == n
    assert len({id(s) for s in seen}) == n  # every search got its own session
    assert state.session not in seen  # never the loop's shared session


def test_analytical_insufficiency_rejects_evidence_free_observation() -> None:
    """A claim with no evidence_chunks is rejected even if other signals look fine."""
    findings = AnalyticalFindings(
        question="q",
        observations=(
            Observation(
                aspect="revenue", claim="Revenue grew", evidence_chunks=[], confidence="high"
            ),
        ),
    )
    reason = gates_module._analytical_insufficiency(findings)
    assert reason is not None
    assert "no evidence_chunks" in reason


def test_stub_rejected_tool_call_strips_arguments() -> None:
    """Rejected finalizer arguments are stubbed so stale claims don't linger in history."""
    from src.services.chat.agent.transcript import stub_rejected_tool_call

    tc = ToolCallRef(
        id="call_1", name="report_analytical_findings", arguments=json.dumps({"claim": "x"})
    )
    stubbed = stub_rejected_tool_call(tc)
    assert stubbed.id == tc.id
    assert stubbed.name == tc.name
    assert "claim" not in stubbed.arguments


def test_drop_evidence_free_observations_moves_claim_to_gaps() -> None:
    """An observation with no evidence is routed into gaps instead of reaching synthesis."""
    findings = AnalyticalFindings(
        question="q",
        observations=(
            Observation(
                aspect="grounded", claim="Grounded claim", evidence_chunks=["c1"], confidence="high"
            ),
            Observation(
                aspect="ungrounded", claim="Ungrounded claim", evidence_chunks=[], confidence="high"
            ),
        ),
    )
    result = gates_module.drop_evidence_free_observations(findings)
    assert [o.claim for o in result.observations] == ["Grounded claim"]
    assert any("Ungrounded claim" in g for g in result.gaps or [])


# ---------------------------------------------------------------------------
# Sequential depth (find-then-follow) — loop-level coverage (T-10)
# ---------------------------------------------------------------------------


def _analytical_state() -> ChatPipelineState:
    """A pipeline state routed `analytical`, which is what loads the v4 prompt and makes
    `named_item_gate` reachable (FR-11)."""
    state = _make_state()
    state.router_output.query_shape = "analytical"  # type: ignore[union-attr]
    return state


def _report_analytical(
    call_id: str,
    *observations: dict[str, Any],
    conclusion: str = "A conclusion.",
) -> ToolCallRef:
    return ToolCallRef(
        id=call_id,
        name="report_analytical_findings",
        arguments=json.dumps(
            {
                "question": "Why did margin compress?",
                "conclusion": conclusion,
                "gaps": [],
                "observations": list(observations),
            }
        ),
    )


def _observation(
    aspect: str,
    claim: str,
    *,
    item: str | None = None,
    status: str = "unresolved",
    ref: str = "S1",
) -> dict[str, Any]:
    obs: dict[str, Any] = {
        "aspect": aspect,
        "claim": claim,
        "evidence_chunks": [ref],
        "confidence": "high",
        "refuted_by": None,
        "named_item": None,
    }
    if item is not None:
        obs["named_item"] = {"name": item, "status": status}
    return obs


def _search(call_id: str, query: str) -> ToolCallRef:
    return ToolCallRef(
        id=call_id,
        name="search_documents",
        arguments=json.dumps({"entity": "Acme", "query": query}),
    )


@pytest.fixture
def _capture_gate_state(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Record the live `AgentRunState` each time `named_item_gate` is consulted.

    The gate is left doing its real work — this only keeps a handle on the state object
    `run_loop` builds internally, which is otherwise unreachable from a caller and is
    where FR-5's counters live.
    """
    seen: list[Any] = []
    real = gates_module.named_item_gate

    def _spy(candidate: Any, run_state: Any) -> Any:
        seen.append(run_state)
        return real(candidate, run_state)

    monkeypatch.setattr(gates_module, "named_item_gate", _spy)
    # tools.py captured the original function object in TOOL_REGISTRY at import time, so
    # the registration must be re-pointed too — and loop.py's `gate is named_item_gate`
    # attribution check compares against the module attribute, which now *is* the spy.
    import src.services.chat.agent.tools as tools_module

    reg = tools_module.TOOL_REGISTRY["report_analytical_findings"]
    monkeypatch.setitem(
        tools_module.TOOL_REGISTRY,
        "report_analytical_findings",
        type(reg)(
            schema=reg.schema,
            terminal=reg.terminal,
            gates=tuple(_spy if g is real else g for g in reg.gates),
        ),
    )
    return seen


@pytest.mark.asyncio
async def test_named_item_reject_then_search_then_resolve(
    monkeypatch: pytest.MonkeyPatch, _capture_gate_state: list[Any]
) -> None:
    """AC-2 -> AC-3 end to end: the loop keeps going after a named-item rejection, the
    model searches, re-reports resolved, and the sealed projection carries the value.

    Also pins the budget-attribution fix: a named-item rejection must leave
    `insufficiency_rejections` at 0, or (default cap 1) it silently disables
    `analytical_insufficiency_gate` for the rest of the run (D1, FR-5).
    """
    monkeypatch.setenv("AGENT_MAX_ITERATIONS", "5")
    state = _analytical_state()
    chunk, payloads = _make_chunk_with_payload()

    adapter = AsyncMock()
    adapter.complete_with_tools = AsyncMock(
        side_effect=[
            AssistantTurnResult(text="", tool_calls=[_search("c0", "margin drivers")]),
            AssistantTurnResult(
                text="",
                tool_calls=[
                    _report_analytical(
                        "c1",
                        _observation(
                            "segment_mix",
                            "Management cited the Payments segment; its margin is not stated here.",
                            item="Payments segment",
                            status="unresolved",
                        ),
                    )
                ],
            ),
            AssistantTurnResult(
                text="", tool_calls=[_search("c2", "Payments segment margin segment_mix")]
            ),
            AssistantTurnResult(
                text="",
                tool_calls=[
                    _report_analytical(
                        "c3",
                        _observation(
                            "segment_mix",
                            "The Payments segment posted a 22% gross margin, below the 41% group average.",
                            item="Payments segment",
                            status="resolved",
                        ),
                    )
                ],
            ),
        ]
    )

    async def _fake_execute_search(*_a: Any, **_k: Any) -> _SearchResult:
        return _SearchResult(entity="Acme", chunks=[chunk], payloads=payloads)

    before = _counter_value("report_analytical_findings", "rejected_named_item")

    _registry, findings, meta = await run_loop(
        state,
        _routed_llm(adapter),
        state.session,
        state.redis_app,
        state.request_id,
        reranker=None,
        session_factory=_fake_session_factory(),
        execute_search=_fake_execute_search,
    )

    # The loop CONTINUED past the rejection rather than stopping or finalizing on it.
    assert meta.iterations == 4
    assert meta.convergence_reason == "natural"
    assert meta.sealed is True

    assert isinstance(findings, AnalyticalFindings)
    assert len(findings.observations) == 1
    resolved = findings.observations[0]
    assert resolved.named_item is not None
    assert resolved.named_item.status == "resolved"
    assert "22%" in resolved.claim

    run_state = _capture_gate_state[-1]
    assert run_state.named_item_rejections == {"payments segment": 1}
    assert run_state.named_item_rejections_total == 1
    # D1 / FR-5: the named-item rejection spent only its own budget.
    assert run_state.insufficiency_rejections == 0

    # D15: the rejection is independently measurable from the existing metric.
    assert _counter_value("report_analytical_findings", "rejected_named_item") == before + 1


def _counter_value(tool: str, status: str) -> float:
    from src.observability.metrics import AGENT_TOOL_CALLS

    return AGENT_TOOL_CALLS.labels(tool, status)._value.get()


@pytest.mark.asyncio
async def test_two_unresolved_items_are_chased_in_one_turn(
    monkeypatch: pytest.MonkeyPatch, _capture_gate_state: list[Any]
) -> None:
    """AC-18, EC-12, D7: two items unresolved at once cost 2 units of the shared pool, and
    both follow-up searches dispatch through the pre-existing `search_sem` — no new limit."""
    monkeypatch.setenv("AGENT_MAX_ITERATIONS", "5")
    monkeypatch.setenv("AGENT_MAX_CONCURRENT_SEARCHES", "2")
    state = _analytical_state()
    chunk, payloads = _make_chunk_with_payload()

    adapter = AsyncMock()
    adapter.complete_with_tools = AsyncMock(
        side_effect=[
            AssistantTurnResult(text="", tool_calls=[_search("c0", "margin drivers")]),
            AssistantTurnResult(
                text="",
                tool_calls=[
                    _report_analytical(
                        "c1",
                        _observation("segment_mix", "Payments named.", item="Payments segment"),
                        _observation("sub_mix", "Aurora named.", item="Aurora Holdings"),
                    )
                ],
            ),
            # Both follow-ups issued in the SAME turn — the mechanism adds no serialization.
            AssistantTurnResult(
                text="",
                tool_calls=[
                    _search("c2", "Payments segment segment_mix"),
                    _search("c3", "Aurora Holdings sub_mix"),
                ],
            ),
            AssistantTurnResult(
                text="",
                tool_calls=[
                    _report_analytical(
                        "c4",
                        _observation(
                            "segment_mix",
                            "Payments margin was 22%.",
                            item="Payments segment",
                            status="resolved",
                        ),
                        _observation(
                            "sub_mix",
                            "Aurora contributed $4m.",
                            item="Aurora Holdings",
                            status="resolved",
                        ),
                    )
                ],
            ),
        ]
    )

    in_flight = 0
    peak = 0

    async def _fake_execute_search(*_a: Any, **_k: Any) -> _SearchResult:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0)
        in_flight -= 1
        return _SearchResult(entity="Acme", chunks=[chunk], payloads=payloads)

    _registry, findings, meta = await run_loop(
        state,
        _routed_llm(adapter),
        state.session,
        state.redis_app,
        state.request_id,
        reranker=None,
        session_factory=_fake_session_factory(),
        execute_search=_fake_execute_search,
    )

    assert meta.sealed is True
    assert isinstance(findings, AnalyticalFindings)
    assert {o.named_item.status for o in findings.observations if o.named_item} == {"resolved"}

    run_state = _capture_gate_state[-1]
    # D7: one rejection carrying two items spends two units, not one.
    assert run_state.named_item_rejections_total == 2
    assert run_state.named_item_rejections == {"payments segment": 1, "aurora holdings": 1}

    # EC-12: bounded by the existing semaphore, and genuinely concurrent within it.
    assert peak == 2


@pytest.mark.asyncio
async def test_empty_and_failed_search_rounds_leave_named_item_counters_untouched(
    monkeypatch: pytest.MonkeyPatch, _capture_gate_state: list[Any]
) -> None:
    """AC-7, FR-10, EC-3: an unproductive round (zero chunks) and an outright search
    failure both flow through the pre-existing empty-round path. Neither is a finalize
    rejection, so neither moves an FR-5 counter."""
    monkeypatch.setenv("AGENT_MAX_ITERATIONS", "5")
    monkeypatch.setenv("AGENT_MAX_CONCURRENT_SEARCHES", "2")
    state = _analytical_state()
    chunk, payloads = _make_chunk_with_payload()

    adapter = AsyncMock()
    adapter.complete_with_tools = AsyncMock(
        side_effect=[
            AssistantTurnResult(text="", tool_calls=[_search("c0", "margin drivers")]),
            # One search returns nothing; the other fails outright (error_str, the shape
            # `_execute_search` produces for a retrieval failure — it does not raise).
            AssistantTurnResult(
                text="",
                tool_calls=[_search("c1", "nothing here"), _search("c2", "boom")],
            ),
            AssistantTurnResult(
                text="",
                tool_calls=[
                    _report_analytical(
                        "c3",
                        _observation(
                            "segment_mix",
                            "Payments margin was 22%.",
                            item="Payments segment",
                            status="resolved",
                        ),
                    )
                ],
            ),
        ]
    )

    async def _fake_execute_search(tc: Any, *_a: Any, **_k: Any) -> _SearchResult:
        query = json.loads(tc.arguments)["query"]
        if query == "nothing here":
            return _SearchResult(entity="Acme", chunks=[], payloads={})
        if query == "boom":
            return _SearchResult(
                entity="Acme", chunks=[], payloads={}, error_str="search failed: upstream 503"
            )
        return _SearchResult(entity="Acme", chunks=[chunk], payloads=payloads)

    before = _counter_value("report_analytical_findings", "rejected_named_item")

    _registry, findings, meta = await run_loop(
        state,
        _routed_llm(adapter),
        state.session,
        state.redis_app,
        state.request_id,
        reranker=None,
        session_factory=_fake_session_factory(),
        execute_search=_fake_execute_search,
    )

    assert meta.sealed is True
    assert isinstance(findings, AnalyticalFindings)

    run_state = _capture_gate_state[-1]
    assert run_state.empty_rounds == 1  # the unproductive round was counted, once
    assert run_state.named_item_rejections == {}
    assert run_state.named_item_rejections_total == 0
    assert run_state.insufficiency_rejections == 0
    assert _counter_value("report_analytical_findings", "rejected_named_item") == before
