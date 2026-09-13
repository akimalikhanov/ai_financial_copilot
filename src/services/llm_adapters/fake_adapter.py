"""A zero-cost, zero-network LLMAdapter for load testing (Stage 17.5 Phase 8).

Wired in via `provider: fake` in a models.yaml config (see infra/config/models.loadtest.yaml),
never reachable from the real infra/config/models.yaml. See docs/notes/loadtest-concepts.md §6
for why: a load test needs realistic LLM *latency* (to exercise queueing/capacity behaviour) but
zero real spend and deterministic tool-calling shape (to keep results reproducible run to run).

The adapter instance is a process-wide singleton (cached inside LLMRouter), shared across
concurrent in-flight requests. Every method below is therefore stateless per call — turn number
in `complete_with_tools` is derived by counting tool messages already in the transcript, never
stored on `self` — so concurrent requests can't cross-contaminate each other's scripted turn.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
from collections.abc import AsyncGenerator, Sequence
from typing import Any
from uuid import uuid4

from src.services.llm_adapters.base_adapter import (
    AssistantTurnResult,
    ChatMessage,
    ChatRequest,
    LLMAdapter,
    LLMResponse,
    LLMResponseStats,
    LLMStreamChunk,
    Role,
    ToolCallRef,
)

# Defaults centered on the measured per-turn LLM latency (mean ~9.3s), see
# docs/notes/capacity-planning-concepts.md §1 "Service time is not a constant". Used for
# complete_with_tools (agent turns) and _stream (final answers) — the heavy calls.
_DEFAULT_LATENCY_MIN_S = 2.0
_DEFAULT_LATENCY_MODE_S = 6.0
_DEFAULT_LATENCY_MAX_S = 20.0

# Structured-output supporting calls (query router, query transformer) are real gpt-4o-mini
# calls that return in ~1-2s, not full agent turns — and query_transformer specifically wraps
# its call in a 10s asyncio.wait_for (QUERY_TRANSFORMER_TIMEOUT, config.py). Sampling from the
# slow profile above regularly exceeded that timeout and spammed rewrite_query_llm_error on
# every load test run (harmless — query_transformer falls back — but it's noise that isn't a
# real finding). Keep this comfortably under the smallest known caller timeout.
_FAST_LATENCY_MIN_S = 0.3
_FAST_LATENCY_MODE_S = 1.0
_FAST_LATENCY_MAX_S = 3.0

# Number of search_documents turns to script before the final report call — the scripted run
# reports once tool_turns_so_far reaches this value. Chosen so total turns (_SEARCH_TURNS + 1
# report turn) lands near the measured 5.51 turns/query average (capacity-planning-concepts.md
# §1). A prior version hardcoded 1 search turn (2 turns total), giving a ~18-20s scripted
# pipeline latency against a real ~40.5s mean. Production's AGENT_MAX_ITERATIONS default (5)
# caps this from above regardless, so this only needs to be a realistic target, not exact.
_SEARCH_TURNS = 4

_FAKE_ANSWER_TEXT = (
    "Based on the retrieved excerpts, the requested figure is available. "
    '<claim refs="S1">This is a synthetic answer produced by the load-test fake LLM adapter.</claim>'
)


async def _sample_latency_seconds(*, fast: bool = False) -> float:
    """Sleep for a realistic-but-fake LLM call duration. Re-read from env on every call —
    the adapter instance is a shared singleton, so this must not be cached on self.

    `fast=True` is for structured-output supporting calls (query router/transformer) — see
    the module-level comment on _FAST_LATENCY_* above. FAKE_LLM_LATENCY_MS still overrides
    both profiles unconditionally — it's the deliberate T7 dependency-slow-LLM knob and
    should be able to force a slow value everywhere, including onto the fast path, when
    someone wants to test what a slow query_transformer call actually does downstream.
    """
    fixed_ms = os.environ.get("FAKE_LLM_LATENCY_MS")
    if fixed_ms is not None:
        delay = float(fixed_ms) / 1000.0
    elif fast:
        low = float(os.environ.get("FAKE_LLM_FAST_LATENCY_MIN_S", _FAST_LATENCY_MIN_S))
        mode = float(os.environ.get("FAKE_LLM_FAST_LATENCY_MODE_S", _FAST_LATENCY_MODE_S))
        high = float(os.environ.get("FAKE_LLM_FAST_LATENCY_MAX_S", _FAST_LATENCY_MAX_S))
        delay = random.triangular(low, high, mode)
    else:
        low = float(os.environ.get("FAKE_LLM_LATENCY_MIN_S", _DEFAULT_LATENCY_MIN_S))
        mode = float(os.environ.get("FAKE_LLM_LATENCY_MODE_S", _DEFAULT_LATENCY_MODE_S))
        high = float(os.environ.get("FAKE_LLM_LATENCY_MAX_S", _DEFAULT_LATENCY_MAX_S))
        delay = random.triangular(low, high, mode)
    await asyncio.sleep(delay)
    return delay


def _last_user_content(messages: Sequence[ChatMessage]) -> str:
    for m in reversed(messages):
        if m.role == Role.user and m.content:
            return m.content
    return "fake adapter: no user message found"


def _fake_stats(*, input_text: str, output_text: str, latency_s: float) -> LLMResponseStats:
    # Rough chars/4 approximation is all a synthetic stat needs to look plausible.
    input_tokens = max(1, len(input_text) // 4)
    output_tokens = max(1, len(output_text) // 4)
    return LLMResponseStats(
        input_tokens=input_tokens,
        cached_input_tokens=0,
        output_tokens=output_tokens,
        reasoning_tokens=0,
        total_tokens=input_tokens + output_tokens,
        latency_ms=latency_s * 1000,
        ttft_ms=min(latency_s * 1000, 500.0),
        tps=output_tokens / latency_s if latency_s > 0 else None,
        cost_usd=0.0,  # the whole point: no real spend, ever
    )


def _response_format_name(req: ChatRequest) -> str | None:
    response_format = req.extra_params.get("response_format")
    if not isinstance(response_format, dict):
        return None
    return response_format.get("json_schema", {}).get("name")


def _canned_text_for(req: ChatRequest) -> str:
    """Structured-output callers (query router, query transformer) parse this with
    json.loads + strict Pydantic validation — see src/services/router/parser.py and
    src/services/retrieval/query_transformer.py::_parse_response. Anything else (naming,
    table summarizer, picture enricher) just needs non-empty text."""
    name = _response_format_name(req)
    if name == "query_router":
        return json.dumps(
            {
                "route": "retrieval",
                "entities": [],
                "user_intent": "fake adapter",
                "reasoning": "fake adapter canned response",
                "query_shape": "extraction",
                "requested_currency": None,
            }
        )
    if name == "query_transformer":
        query = _last_user_content(req.messages)
        return json.dumps({"semantic_query": query, "keyword_query": query, "fallback": False})
    return "Fake adapter canned response for load testing."


class FakeAdapter(LLMAdapter):
    """No network calls, no cost, sampled or fixed latency. See module docstring."""

    provider_name = "fake"

    def __init__(self, *, default_model: str):
        super().__init__(default_model=default_model)

    async def close(self) -> None:
        pass

    async def _complete(self, req: ChatRequest) -> LLMResponse:
        # response_format present => a structured-output supporting call (query router/
        # transformer), not a full agent turn or answer — see _FAST_LATENCY_* above.
        is_fast_call = _response_format_name(req) is not None
        latency_s = await _sample_latency_seconds(fast=is_fast_call)
        text = _canned_text_for(req)
        stats = _fake_stats(
            input_text=_last_user_content(req.messages), output_text=text, latency_s=latency_s
        )
        return LLMResponse(text=text, raw=None, stats=stats)

    async def _stream(self, req: ChatRequest) -> AsyncGenerator[LLMStreamChunk, None]:
        latency_s = await _sample_latency_seconds()
        words = _FAKE_ANSWER_TEXT.split(" ")
        for word in words[:-1]:
            yield LLMStreamChunk(text=word + " ", raw=None, is_final=False)
        stats = _fake_stats(
            input_text=_last_user_content(req.messages),
            output_text=_FAKE_ANSWER_TEXT,
            latency_s=latency_s,
        )
        yield LLMStreamChunk(text=words[-1], raw=None, is_final=True, stats=stats)

    async def complete_with_tools(
        self,
        messages: Sequence[ChatMessage],
        tools: list[dict[str, Any]],
        **_kwargs: Any,
    ) -> AssistantTurnResult:
        latency_s = await _sample_latency_seconds()
        tool_turns_so_far = sum(1 for m in messages if m.role == Role.tool)
        tool_names = {t.get("function", {}).get("name") for t in tools}

        if tool_turns_so_far < _SEARCH_TURNS:
            args = json.dumps(
                {"entity": "Fake Entity", "query": _last_user_content(messages)[:200]}
            )
            tool_calls = [
                ToolCallRef(id=f"fake-{uuid4()}", name="search_documents", arguments=args)
            ]
            text = ""
        elif tool_turns_so_far == _SEARCH_TURNS:
            if "report_analytical_findings" in tool_names:
                args = json.dumps(
                    {
                        "question": _last_user_content(messages)[:200],
                        "observations": [
                            {
                                "aspect": "A1",
                                "claim": "Fake adapter synthetic observation for load testing.",
                                "substantiated": True,
                                "evidence_chunks": ["S1"],
                                "confidence": "medium",
                            }
                        ],
                        "conclusion": None,
                    }
                )
                report_name = "report_analytical_findings"
            else:
                args = json.dumps(
                    {
                        "metric_requested": "fake metric",
                        "findings": [
                            {
                                "entity": "Fake Entity",
                                "available": True,
                                "value": 1.0,
                                "currency": "USD",
                                "source_chunks": ["S1"],
                                "unit": "",
                            }
                        ],
                    }
                )
                report_name = "report_findings"
            tool_calls = [ToolCallRef(id=f"fake-{uuid4()}", name=report_name, arguments=args)]
            text = ""
        else:
            tool_calls = []
            text = "Fake adapter: no further tool calls."

        stats = _fake_stats(
            input_text=_last_user_content(messages), output_text=text, latency_s=latency_s
        )
        return AssistantTurnResult(text=text, tool_calls=tool_calls, stats=stats)
