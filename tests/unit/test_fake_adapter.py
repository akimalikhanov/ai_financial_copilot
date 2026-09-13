"""FakeAdapter is a real LLMAdapter shipped for the Stage 17.5 Phase 8 load test, not a test
mock — see src/services/llm_adapters/fake_adapter.py and docs/notes/loadtest-concepts.md §6.
These tests prove it's safe to deploy: it must never make a real call, must never crash the
callers that parse structured output, and must not leak state across concurrent requests
despite being a process-wide singleton (see llm_router.py::_build_adapter / get_router).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from src.schemas.query_router import RouterOutput
from src.schemas.query_transform import TransformedQuery
from src.services.chat.agent.loop import _SearchResult, run_loop
from src.services.llm_adapters import fake_adapter as fake_adapter_module
from src.services.llm_adapters.base_adapter import ChatMessage, ChatRequest, Role
from src.services.llm_adapters.fake_adapter import FakeAdapter
from src.services.llm_router import _build_adapter
from src.services.llm_runtime.exceptions import LLMServerError
from src.services.router.parser import parse_router_response
from src.utils.config import load_models_config
from src.utils.json_schema import build_response_format

from .test_agent_loop_smoke import (
    _fake_session_factory,
    _make_chunk_with_payload,
    _make_state,
    _routed_llm,
)


def _user_message(text: str) -> ChatMessage:
    return ChatMessage(role=Role.user, content=text)


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests fast: record the requested delay instead of actually sleeping."""

    async def _instant_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("asyncio.sleep", _instant_sleep)


def test_build_adapter_returns_fake_adapter() -> None:
    adapter = _build_adapter("fake", {"model_name": "fake-test-model"})
    assert isinstance(adapter, FakeAdapter)
    assert adapter.default_model == "fake-test-model"


@pytest.mark.parametrize("provider", ["openai", "google", "vllm"])
def test_fake_llm_only_refuses_real_providers(
    provider: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The load test's last line of defence. A pod that races the config rollout can still be
    holding the production models.yaml; refusing to build the adapter is what stops it from
    billing a real provider."""
    monkeypatch.setenv("FAKE_LLM_ONLY", "true")
    with pytest.raises(LLMServerError, match="FAKE_LLM_ONLY"):
        _build_adapter(provider, {"model_name": "gpt-4o-mini", "model_path": "/m"})


def test_fake_llm_only_still_allows_the_fake_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_LLM_ONLY", "true")
    assert isinstance(_build_adapter("fake", {"model_name": "fake-test-model"}), FakeAdapter)


def test_real_providers_build_when_the_guard_is_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard must be opt-in — production sets no such variable."""
    monkeypatch.delenv("FAKE_LLM_ONLY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-a-real-key")
    assert not isinstance(_build_adapter("openai", {"model_name": "gpt-4o-mini"}), FakeAdapter)


def test_models_config_path_accepts_a_relative_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """A relative MODELS_CONFIG_PATH resolves against the project root, so the same value works
    on the host and inside a container (where the repo lives at /app). An absolute container
    path in .env broke every test in the suite once — pytest loads .env, and /app doesn't
    exist on the host."""
    monkeypatch.setenv("MODELS_CONFIG_PATH", "infra/config/models.loadtest.yaml")
    config = load_models_config()
    assert {m["provider"] for m in config["models"]} == {"fake"}


def test_models_config_path_unset_loads_the_real_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """The override must be inert when unset — that's what keeps every real environment on
    infra/config/models.yaml with zero behavior change."""
    monkeypatch.delenv("MODELS_CONFIG_PATH", raising=False)
    config = load_models_config()
    assert "fake" not in {m["provider"] for m in config["models"]}


@pytest.mark.asyncio
async def test_complete_query_router_response_parses() -> None:
    adapter = FakeAdapter(default_model="fake")
    response_format = build_response_format("query_router", RouterOutput.model_json_schema())
    req = ChatRequest(
        messages=(_user_message("What was Acme's revenue?"),),
        model="fake",
        extra_params={"response_format": response_format},
    )
    resp = await adapter._complete(req)
    parsed, error = parse_router_response(resp.text)
    assert error is None
    assert parsed is not None
    assert parsed.route == "retrieval"
    assert resp.stats is not None
    assert resp.stats.cost_usd == 0.0


@pytest.mark.asyncio
async def test_complete_query_transformer_response_parses() -> None:
    adapter = FakeAdapter(default_model="fake")
    response_format = build_response_format(
        "query_transformer", TransformedQuery.model_json_schema()
    )
    req = ChatRequest(
        messages=(_user_message("What was Acme's revenue?"),),
        model="fake",
        extra_params={"response_format": response_format},
    )
    resp = await adapter._complete(req)
    parsed = TransformedQuery.model_validate(json.loads(resp.text))
    assert parsed.semantic_query
    assert parsed.fallback is False


@pytest.mark.asyncio
async def test_complete_generic_response_is_nonempty_text() -> None:
    adapter = FakeAdapter(default_model="fake")
    req = ChatRequest(messages=(_user_message("hello"),), model="fake")
    resp = await adapter._complete(req)
    assert resp.text
    assert resp.stats is not None and resp.stats.cost_usd == 0.0


@pytest.mark.asyncio
async def test_stream_yields_final_chunk_with_stats() -> None:
    adapter = FakeAdapter(default_model="fake")
    req = ChatRequest(messages=(_user_message("hello"),), model="fake")
    chunks = [c async for c in adapter._stream(req)]
    assert chunks
    assert chunks[-1].is_final
    assert chunks[-1].stats is not None
    # Non-final chunks carry no stats — mirrors OpenAIAdapter's shape.
    assert all(not c.is_final for c in chunks[:-1])


@pytest.mark.asyncio
async def test_complete_with_tools_scripts_search_then_report_then_stop() -> None:
    """Scripts _SEARCH_TURNS search_documents calls, then one report_findings call, then stops —
    see fake_adapter._SEARCH_TURNS for why the search count is 4, not 1."""
    adapter = FakeAdapter(default_model="fake")
    tools: list[dict[str, Any]] = [
        {"type": "function", "function": {"name": "search_documents"}},
        {"type": "function", "function": {"name": "report_findings"}},
    ]
    messages = [_user_message("What was Acme's revenue?")]

    for _ in range(fake_adapter_module._SEARCH_TURNS):
        turn = await adapter.complete_with_tools(messages, tools)
        assert len(turn.tool_calls) == 1
        assert turn.tool_calls[0].name == "search_documents"
        json.loads(turn.tool_calls[0].arguments)  # must be valid JSON
        messages = [
            *messages,
            ChatMessage(role=Role.tool, tool_call_id=turn.tool_calls[0].id, content="found stuff"),
        ]

    report_turn = await adapter.complete_with_tools(messages, tools)
    assert len(report_turn.tool_calls) == 1
    assert report_turn.tool_calls[0].name == "report_findings"
    json.loads(report_turn.tool_calls[0].arguments)

    messages_after_report = [
        *messages,
        ChatMessage(role=Role.tool, tool_call_id=report_turn.tool_calls[0].id, content="recorded"),
    ]
    final_turn = await adapter.complete_with_tools(messages_after_report, tools)
    assert final_turn.tool_calls == []


@pytest.mark.asyncio
async def test_complete_with_tools_uses_analytical_report_when_offered() -> None:
    adapter = FakeAdapter(default_model="fake")
    analytical_tools: list[dict[str, Any]] = [
        {"type": "function", "function": {"name": "search_documents"}},
        {"type": "function", "function": {"name": "report_analytical_findings"}},
    ]
    messages = [
        _user_message("Compare Acme and Beta"),
        *(
            ChatMessage(role=Role.tool, tool_call_id=f"call_{i}", content="found stuff")
            for i in range(fake_adapter_module._SEARCH_TURNS)
        ),
    ]
    turn = await adapter.complete_with_tools(messages, analytical_tools)
    assert turn.tool_calls[0].name == "report_analytical_findings"
    json.loads(turn.tool_calls[0].arguments)


@pytest.mark.asyncio
async def test_complete_with_tools_turn_is_derived_from_transcript_not_shared_state() -> None:
    """The adapter is a process-wide singleton (see llm_router.py::get_router) shared across
    concurrent requests. Interleaving two independent conversations on the same instance must
    not cross-contaminate which turn each one is on."""
    adapter = FakeAdapter(default_model="fake")
    tools: list[dict[str, Any]] = [
        {"type": "function", "function": {"name": "search_documents"}},
        {"type": "function", "function": {"name": "report_findings"}},
    ]
    conversation_a_turn0 = [_user_message("Question A")]
    conversation_b_at_report_turn = [
        _user_message("Question B"),
        *(
            ChatMessage(role=Role.tool, tool_call_id=f"b{i}", content="found stuff")
            for i in range(fake_adapter_module._SEARCH_TURNS)
        ),
    ]

    # Call B's report turn first, then A's turn 0 — order must not matter.
    result_b = await adapter.complete_with_tools(conversation_b_at_report_turn, tools)
    result_a = await adapter.complete_with_tools(conversation_a_turn0, tools)

    assert result_b.tool_calls[0].name == "report_findings"
    assert result_a.tool_calls[0].name == "search_documents"


@pytest.mark.asyncio
async def test_structured_output_calls_stay_under_query_transformer_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """query_transformer.py wraps its LLM call in a 10s asyncio.wait_for
    (QUERY_TRANSFORMER_TIMEOUT, config.py). Sampling from the slow ~2-20s agent-turn profile
    for this call regularly raced that timeout and spammed rewrite_query_llm_error on every
    load test run (harmless — it falls back — but noise, not a real finding). Confirmed by an
    actual load test run on 2026-09-09."""
    recorded: list[float] = []

    async def _recording_sleep(seconds: float) -> None:
        recorded.append(seconds)

    monkeypatch.setattr("asyncio.sleep", _recording_sleep)
    adapter = FakeAdapter(default_model="fake")
    response_format = build_response_format(
        "query_transformer", TransformedQuery.model_json_schema()
    )
    req = ChatRequest(
        messages=(_user_message("hello"),),
        model="fake",
        extra_params={"response_format": response_format},
    )
    for _ in range(50):
        await adapter._complete(req)
    assert all(delay < 10.0 for delay in recorded)


@pytest.mark.asyncio
async def test_fixed_latency_env_var_overrides_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: list[float] = []

    async def _recording_sleep(seconds: float) -> None:
        recorded.append(seconds)

    monkeypatch.setattr("asyncio.sleep", _recording_sleep)
    monkeypatch.setenv("FAKE_LLM_LATENCY_MS", "20000")

    adapter = FakeAdapter(default_model="fake")
    req = ChatRequest(messages=(_user_message("hello"),), model="fake")
    await adapter._complete(req)

    assert recorded == [20.0]


@pytest.mark.asyncio
async def test_run_loop_terminates_cleanly_with_real_fake_adapter() -> None:
    """End-to-end: wire a real FakeAdapter (not AsyncMock) into run_loop and confirm the
    agent loop's state machine — turn partitioning, tool-result bookkeeping, termination —
    tolerates it, the same way it tolerates a real provider."""
    state = _make_state()
    found_chunk, payloads = _make_chunk_with_payload()
    adapter = FakeAdapter(default_model="fake")
    llm = _routed_llm(adapter)

    async def _fake_execute_search(*_args: Any, **_kwargs: Any) -> _SearchResult:
        return _SearchResult(entity="Fake Entity", chunks=[found_chunk], payloads=payloads)

    evidence, agent_findings, meta = await run_loop(
        state,
        llm,
        state.session,
        state.redis_app,
        state.request_id,
        reranker=None,
        session_factory=_fake_session_factory(),
        execute_search=_fake_execute_search,
    )

    assert meta.iterations <= 3
    assert meta.convergence_reason is not None
