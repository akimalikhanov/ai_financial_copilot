"""Agent state: settings, per-request effort prior, and the run state the loop mutates.

Two kinds of state (see docs/stages/agentic_state_refactor_v2.md): `transcript` is the
model's view (lossy, compactable); `evidence` + the record fields below are the durable,
never-truncated record. `AgentRunState` is neither alone — it is transcript + record +
loop control.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from src.services.chat.agent.evidence import EvidenceLedger
from src.services.chat.agent.findings import FindingsLedger
from src.services.chat.agent.transcript import Transcript
from src.utils.config import get_query_transformer_model

if TYPE_CHECKING:
    from src.services.llm_adapters.base_adapter import LLMResponseStats

ConvergenceReason = Literal["natural", "convergence", "iteration_cap", "budget_cap", "timeout"]


class AgentSettings(BaseModel):
    """Validated agent env config (P2-15) — replaces the untyped dict from get_agent_config()."""

    tool_model: str
    max_iterations: int = Field(ge=1, le=20)
    token_budget: int = Field(ge=1000)
    max_concurrent_searches: int = Field(ge=1, le=16)
    max_chunks_per_entity: int = Field(ge=1)
    max_empty_analytical_rounds: int = Field(ge=0)
    # Cap on how many times the thin-analytical-finalizer gate (3.ii/3.iii) may reject
    # and force a re-prompt per request — uncapped rejection loops drove iteration_cap /
    # high token spend with little correctness gain.
    max_insufficiency_rejections: int = Field(ge=0)
    turn_timeout_seconds: float = Field(gt=0)
    # Stage 1.5: per-shape override for analytical queries, which tend to need more
    # search turns to corroborate/refute multiple hypotheses. Defaults to max_iterations
    # so the prior is shape-invariant until an operator tunes it against eval data.
    max_iterations_analytical: int = Field(ge=1, le=20)


def get_agent_settings() -> AgentSettings:
    """Read + validate agent config from env. Not cached — called once per request, like
    the dict it replaces, so env overrides (incl. in tests) always take effect."""
    max_iterations = int(os.getenv("AGENT_MAX_ITERATIONS", "5"))
    return AgentSettings(
        tool_model=os.getenv("AGENT_TOOL_MODEL", get_query_transformer_model()),
        max_iterations=max_iterations,
        token_budget=int(os.getenv("AGENT_TOKEN_BUDGET", "150000")),
        max_concurrent_searches=int(os.getenv("AGENT_MAX_CONCURRENT_SEARCHES", "3")),
        max_chunks_per_entity=int(os.getenv("AGENT_MAX_CHUNKS_PER_ENTITY", "5")),
        max_empty_analytical_rounds=int(os.getenv("AGENT_MAX_EMPTY_ANALYTICAL_ROUNDS", "1")),
        max_insufficiency_rejections=int(os.getenv("AGENT_MAX_INSUFFICIENCY_REJECTIONS", "1")),
        turn_timeout_seconds=float(os.getenv("AGENT_TURN_TIMEOUT_SECONDS", "60")),
        max_iterations_analytical=int(
            os.getenv("AGENT_MAX_ITERATIONS_ANALYTICAL", str(max_iterations))
        ),
    )


@dataclass(frozen=True)
class EffortPrior:
    """Per-request effort budget, derived from (settings, query_shape) — Stage 1.5's
    "soft prior on effort". Gates and the loop read these, never `settings` directly,
    so there is exactly one path from query_shape to effort (doc's "one reader path").
    """

    max_iterations: int
    max_empty_rounds: int
    max_insufficiency_rejections: int
    max_concurrent_searches: int

    @classmethod
    def for_shape(cls, settings: AgentSettings, shape: str | None) -> EffortPrior:
        max_iterations = (
            settings.max_iterations_analytical if shape == "analytical" else settings.max_iterations
        )
        return cls(
            max_iterations=max_iterations,
            max_empty_rounds=settings.max_empty_analytical_rounds,
            max_insufficiency_rejections=settings.max_insufficiency_rejections,
            max_concurrent_searches=settings.max_concurrent_searches,
        )


@dataclass
class TokenSpend:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class AgentRunState:
    """The whole loop state: transcript (view) + record (durable) + control.

    `state.transcript` beside `state.evidence` reads as view-vs-record at every call
    site — that grouping is the point (see docs *The two kinds of state*).
    """

    # --- input: not state, but read throughout the loop ---
    effort: EffortPrior
    token_budget: int
    turn_timeout_seconds: float

    # --- transcript: the model's view. Lossy, compactable, never authoritative. ---
    transcript: Transcript

    # --- durable record: complete, never truncated ---
    evidence: EvidenceLedger = field(default_factory=EvidenceLedger)
    findings: FindingsLedger = field(default_factory=FindingsLedger)  # what we concluded (keyed)
    sealed: bool = False  # did a finalizer commit the ledger (vs a degraded projection)
    expected_entities: set[str] = field(default_factory=set)
    searched_entities: set[str] = field(default_factory=set)
    spend: dict[str, TokenSpend] = field(default_factory=dict)

    # --- control: loop counters + outcome ---
    iteration: int = 0
    empty_rounds: int = 0
    insufficiency_rejections: int = 0
    tool_calls_total: int = 0
    convergence_reason: ConvergenceReason = "iteration_cap"

    def record_spend(self, model_id: str, stats: LLMResponseStats | None) -> None:
        if stats is None:
            return
        ts = self.spend.setdefault(model_id, TokenSpend())
        ts.input_tokens += stats.input_tokens or 0
        ts.output_tokens += stats.output_tokens or 0
        ts.cost_usd += stats.cost_usd or 0.0

    def input_tokens_total(self) -> int:
        return sum(ts.input_tokens for ts in self.spend.values())

    def spend_within_budget(self) -> bool:
        return self.input_tokens_total() <= self.token_budget


@dataclass
class AgentLoopMeta:
    iterations: int
    tool_calls_total: int
    convergence_reason: ConvergenceReason
    # Did a finalizer commit the ledger? False means the served findings are a degraded
    # projection of accumulated partial findings (non-converged run) — the trace marker
    # that tells a degraded serve apart from a sealed one.
    sealed: bool = False
    input_tokens_total: int = 0
    output_tokens_total: int = 0
    cost_usd_total: float = 0.0
    # P0-4: input tokens attributed per model_id (agent tool model vs query-rewrite model).
    # input_tokens_total is their sum; the budget cap checks the sum, unchanged.
    # Entities the loop actually called search_documents for — the synthesis boundary uses
    # this (not reported coverage) to label stubs for entities the agent never searched.
    searched_entities: frozenset[str] = field(default_factory=frozenset)


def debug_snapshot(state: AgentRunState) -> dict:
    """Compact, JSON-safe view across the three state stores (Transcript, EvidenceLedger,
    FindingsLedger), for attaching to a Langfuse span as debugging metadata.

    Without this, the ledgers are invisible in tracing — a rejection or a degraded finalize
    can only be explained by manually replaying every tool-call span in order. Deliberately
    a snapshot, not a store-specific export: callers attach it at the few moments the
    ledgers' combined state actually explains something (a gate rejection, a finalizer
    accept, the run's terminal state), not on every turn.
    """
    findings = state.findings.projection()
    return {
        "iteration": state.iteration,
        "sealed": state.sealed,
        "convergence_reason": state.convergence_reason,
        "findings": findings.model_dump(mode="json") if findings is not None else None,
        "findings_keys": sorted(state.findings.keys()),
        "evidence_chunk_count": len(state.evidence),
        "transcript_message_count": len(state.transcript.messages),
        "searched_entities": sorted(state.searched_entities),
        "expected_entities": sorted(state.expected_entities),
        "empty_rounds": state.empty_rounds,
        "insufficiency_rejections": state.insufficiency_rejections,
        "tool_calls_total": state.tool_calls_total,
    }


def build_meta(state: AgentRunState, iterations: int) -> AgentLoopMeta:
    return AgentLoopMeta(
        iterations=iterations,
        tool_calls_total=state.tool_calls_total,
        convergence_reason=state.convergence_reason,
        sealed=state.sealed,
        input_tokens_total=state.input_tokens_total(),
        output_tokens_total=sum(ts.output_tokens for ts in state.spend.values()),
        cost_usd_total=sum(ts.cost_usd for ts in state.spend.values()),
        searched_entities=frozenset(state.searched_entities),
    )
