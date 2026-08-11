"""Agent state: settings, per-request effort prior, and the run state the loop mutates.

Two kinds of state (see docs/stages/agentic_state_refactor_v2.md): `transcript` is the
model's view (lossy, compactable); `evidence` + the record fields below are the durable,
never-truncated record. `AgentRunState` is neither alone — it is transcript + record +
loop control.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from src.services.chat.agent.evidence import EvidenceLedger
from src.services.chat.agent.findings import FindingsLedger
from src.services.chat.agent.transcript import Transcript
from src.utils.config import get_query_transformer_model

if TYPE_CHECKING:
    from src.services.llm_adapters.base_adapter import LLMResponseStats

ConvergenceReason = Literal[
    "natural",
    "convergence",
    "iteration_cap",
    "budget_cap",
    "timeout",
    # D3/step 3: the loop, not a terminal tool call, now decides when a run is done.
    "covered",  # every planned aspect/entity was reported on
    "search_unavailable",  # every search this turn errored — a dead backend, not an empty corpus
    "deadline",  # wall-clock bound for the whole run (turn_timeout_seconds bounds only one turn)
]


class AgentSettings(BaseModel):
    """Validated agent env config (P2-15) — replaces the untyped dict from get_agent_config()."""

    tool_model: str
    max_iterations: int = Field(ge=1, le=20)
    token_budget: int = Field(ge=1000)
    max_concurrent_searches: int = Field(ge=1, le=16)
    max_chunks_per_entity: int = Field(ge=1)
    max_empty_analytical_rounds: int = Field(ge=0)
    turn_timeout_seconds: float = Field(gt=0)
    # Wall-clock bound on the whole run. `turn_timeout_seconds` bounds a single turn, so
    # without this a run of slow-but-not-timing-out turns has no bound at all.
    deadline_seconds: float = Field(gt=0)
    # Stage 1.5: per-shape override for analytical queries, which tend to need more
    # search turns to corroborate/refute multiple hypotheses. Defaults to 8 — coverage-
    # driven termination needs headroom above the extraction path's default of 5.
    max_iterations_analytical: int = Field(ge=1, le=20)
    # 10b §4: prior conversation was permanent and token-unbounded in the agent
    # transcript (≤50 messages, up to ~10-20k tokens on every turn). Cap it.
    history_turns: int = Field(ge=0)
    history_assistant_tokens: int = Field(ge=1)
    # Ceiling on loop-minted plan entries. Near-duplicate sub_questions each mint their
    # own id (no fuzzy matching), so this is what bounds the cost of that choice.
    max_plan_items: int = Field(ge=1)
    # Per-turn cap on revivals, so a search re-returning a large evicted set cannot
    # reinflate what transcript compaction just shrank.
    max_revivals_per_turn: int = Field(ge=0)


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
        turn_timeout_seconds=float(os.getenv("AGENT_TURN_TIMEOUT_SECONDS", "60")),
        deadline_seconds=float(os.getenv("AGENT_DEADLINE_SECONDS", "180")),
        max_iterations_analytical=int(os.getenv("AGENT_MAX_ITERATIONS_ANALYTICAL", "7")),
        history_turns=int(os.getenv("AGENT_HISTORY_TURNS", "2")),
        history_assistant_tokens=int(os.getenv("AGENT_HISTORY_ASSISTANT_TOKENS", "600")),
        max_plan_items=int(os.getenv("AGENT_MAX_PLAN_ITEMS", "6")),
        max_revivals_per_turn=int(os.getenv("AGENT_MAX_REVIVALS_PER_TURN", "3")),
    )


@dataclass(frozen=True)
class EffortPrior:
    """Per-request effort budget, derived from (settings, query_shape) — Stage 1.5's
    "soft prior on effort". Gates and the loop read these, never `settings` directly,
    so there is exactly one path from query_shape to effort (doc's "one reader path").
    """

    max_iterations: int
    max_empty_rounds: int
    max_concurrent_searches: int
    max_plan_items: int
    max_revivals_per_turn: int
    max_chunks_per_lookup: int

    @classmethod
    def for_shape(cls, settings: AgentSettings, shape: str | None) -> EffortPrior:
        max_iterations = (
            settings.max_iterations_analytical if shape == "analytical" else settings.max_iterations
        )
        return cls(
            max_iterations=max_iterations,
            max_empty_rounds=settings.max_empty_analytical_rounds,
            max_concurrent_searches=settings.max_concurrent_searches,
            max_plan_items=settings.max_plan_items,
            max_revivals_per_turn=settings.max_revivals_per_turn,
            max_chunks_per_lookup=settings.max_chunks_per_entity,
        )


@dataclass
class TokenSpend:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class AspectStats:
    """Per-aspect search-attempt log, written in the turn's reduce loop.

    Lives here rather than on the EvidenceLedger because a failed or empty search admits
    no chunks and so leaves no ledger trace at all — the very case D6 must distinguish.
    `errored == searches` means the backend was unreachable; `searches > 0` with
    `new_chunks == 0` means the corpus genuinely lacks it.
    """

    searches: int = 0
    errored: int = 0
    new_chunks: int = 0


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
    # Set when Stop("covered") closes every planned key. Read `sealed`, not this — an
    # empty plan can never set it, and would otherwise report "didn't finish covering".
    sealed_by_coverage: bool = False
    expected_entities: set[str] = field(default_factory=set)
    searched_entities: set[str] = field(default_factory=set)
    spend: dict[str, TokenSpend] = field(default_factory=dict)

    # --- decomposition plan: loop-minted aspect ids, same shape/role as expected_entities ---
    plan: dict[str, str] = field(default_factory=dict)  # "A1" -> sub_question, insertion-ordered
    reported_keys: set[str] = field(default_factory=set)  # raw keys the model reported on
    aspect_stats: dict[str, AspectStats] = field(default_factory=dict)

    # --- control: loop counters + outcome ---
    iteration: int = 0
    empty_rounds: int = 0
    tool_calls_total: int = 0
    convergence_reason: ConvergenceReason = "iteration_cap"
    # Wall-clock bound for the whole run. `turn_timeout_seconds` bounds one turn, so
    # without this a run of slow-but-not-timing-out turns has no bound at all.
    deadline_seconds: float = 180.0
    started_at: float = field(default_factory=perf_counter)

    # --- step 9 instrumentation ---
    report_calls_total: int = 0
    turns_to_first_report: int | None = None
    unknown_aspect_keys: int = 0
    ungrounded_closes: int = 0
    # Coverage as of the loop's last turn. Step 8 closes every remaining key with a gap,
    # so measuring after that point would report 100% coverage on every run.
    plan_covered_at_stop: int | None = None

    @property
    def addressed(self) -> set[str]:
        """Keys that have actually produced output (D4) — a grounded finding or a stated gap.

        Derived, never stored. 10b step 2 closes an aspect from the *raw* reported keys so
        an unanswerable one doesn't get hammered to budget death, but `findings` only holds
        *grounded* keys. Storing both invites them to disagree, and on disagreement
        `Stop("covered")` would set `sealed=True` while serving an aspect that produced no
        finding and no gap. Deriving makes that unrepresentable: an ungrounded report still
        closes its key, but only via `drop_evidence_free_observations` writing the gap that
        `closed_as_gap` records.
        """
        return self.findings.keys() | self.findings.closed_as_gap()

    @property
    def sealed(self) -> bool:
        """Did the run finish what it set out to cover (vs a degraded projection)?

        A plan is empty in two legitimate cases — extraction whose entities resolved to no
        documents, and an analytical run whose searches carried `sub_question: null` — and
        neither means the run fell short. Reading `sealed_by_coverage` alone conflates
        "nothing to cover" with "didn't finish covering" (P2-I), biasing the
        `agent_findings_sealed` metric low and appending a "did not fully converge" caveat
        to answers that were complete.
        """
        return not self.plan or self.sealed_by_coverage

    def unaccounted_keys(self) -> set[str]:
        """Reported keys that produced neither a finding nor a gap — a debug signal.

        Non-empty means a report was ingested but vanished (an off-kind `record`, or refs
        that resolved to nothing). These keys stay *open*: reconciling them into gaps
        mid-loop would close them → `addressed` → `Stop("covered")` → `sealed`, serving an
        aspect that produced no output at the trust level of a converged run. Left open,
        the aspect stays searchable and the end-of-run sweep gaps it after
        `plan_covered_at_stop` is snapshotted.
        """
        return self.reported_keys - self.addressed

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

    def past_deadline(self) -> bool:
        """Wall-clock bound for the whole run, checked at the end of each turn."""
        return (perf_counter() - self.started_at) >= self.deadline_seconds


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
    # Step 9: decomposition width/coverage and whether reporting was incremental. The
    # kill criterion is report_calls_total ≈ 1 *and* plan_covered/plan_seeded no better
    # than v3 — that means the mechanism is inert while costing an extra tool call.
    plan_seeded: int = 0
    plan_covered: int = 0
    report_calls_total: int = 0
    turns_to_first_report: int | None = None
    unknown_aspect_keys: int = 0
    ungrounded_close_rate: float = 0.0
    # Tracked, never a kill criterion: a run that concludes each aspect once, correctly, is
    # a success. Zero means reporting only ever appends — a revision needs later evidence
    # to contradict an earlier conclusion.
    revised_keys: int = 0


def open_aspects(state: AgentRunState) -> list[str]:
    """Plan entries that have not yet produced a finding or a stated gap, in mint order."""
    addressed = state.addressed
    return [a for a in state.plan if a not in addressed]


def turn_snapshot(state: AgentRunState) -> dict:
    """Coverage state as of *after* this turn's effects landed — structured, not prose.

    A per-turn counterpart to `debug_snapshot`: the same underlying fields (`plan`,
    `addressed`, `open_aspects`, `aspect_stats`), but cheap enough to attach to every
    `agent_turn_N` span rather than only once at the run's end. Without this, Langfuse
    shows each turn's *input* status (coverage before the turn ran, as the prose the model
    read) but nothing structured about what changed after — reconstructing how plan
    coverage evolved turn-over-turn means replaying tool-call args and report outputs by
    hand. Field names deliberately match `debug_snapshot` so the two never drift apart.
    """
    addressed = state.addressed
    return {
        "plan": dict(state.plan),
        "addressed": sorted(addressed),
        "open_aspects": open_aspects(state),
        "aspect_stats": {a: vars(s) for a, s in state.aspect_stats.items()},
        "empty_rounds": state.empty_rounds,
        "reported_keys": sorted(state.reported_keys),
    }


def render_status(state: AgentRunState) -> str | None:
    """Coverage status, computed per call and appended last — never stored (10b §4a).

    Keys and sub-questions only, never claim bodies: re-injecting bodies is the reverted
    commit's "established findings block", whose only purpose was giving the model
    something to copy during a coerced restatement. Being a computed view makes it the
    single source of coverage truth — the report result's own open list goes stale, and
    this wins by recency. The stall nudge lives here too rather than being appended once
    and never removed.
    """
    if not state.plan:
        return None
    addressed = state.addressed
    recorded = [a for a in state.plan if a in addressed]
    still_open = [a for a in state.plan if a not in addressed]
    parts: list[str] = []
    if recorded:
        parts.append("Recorded: " + ", ".join(recorded))
    if still_open:
        parts.append("Open: " + ", ".join(f"{a} ({state.plan[a]})" for a in still_open))
    if state.empty_rounds:
        parts.append(
            "The last search returned no new evidence. The drivers you need are likely in "
            "a different section (a footnote, reconciliation, or segment table) — "
            "reformulate with terms targeting where the magnitudes are disclosed."
        )
    # Deliberately no "you have evidence but recorded nothing" nudge. Tried and reverted:
    # turn 1 with nothing recorded is the normal state of a run about to drill down, so it
    # traded the second search pass for an early report (traces af7363d9 vs df7654f6).
    # Final turn: search tools are withheld, so the only move left is converting admitted
    # evidence into observations. Deliberately routes the pressure into `substantiated:
    # false` rather than into closure — a coerced substantiated claim citing a real but
    # irrelevant label passes `_is_grounded`, closes its aspect, and silently promotes an
    # iteration-capped run to `sealed`, dropping the "did not fully converge" caveat.
    if state.iteration == state.effort.max_iterations - 1:
        parts.append(
            "Final turn — no further searches will run. Report every open aspect from the "
            "evidence you have already retrieved. If an aspect is not supported by what "
            "you have, report it with `substantiated: false` and state the absence in "
            "`claim`. Anything left unreported is recorded as unresolved."
        )
    return " · ".join(parts) if parts else None


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
        "plan": dict(state.plan),
        "addressed": sorted(state.addressed),
        "open_aspects": open_aspects(state),
        "unaccounted_keys": sorted(state.unaccounted_keys()),
        "aspect_stats": {a: vars(s) for a, s in state.aspect_stats.items()},
        "empty_rounds": state.empty_rounds,
        "tool_calls_total": state.tool_calls_total,
        # Step 9: is reporting incremental, or is the model one-shotting anyway?
        "plan_seeded": len(state.plan),
        "plan_covered": (
            state.plan_covered_at_stop
            if state.plan_covered_at_stop is not None
            else len(state.plan.keys() & state.addressed)
        ),
        "report_calls_total": state.report_calls_total,
        "turns_to_first_report": state.turns_to_first_report,
        "unknown_aspect_keys": state.unknown_aspect_keys,
        "ungrounded_closes": state.ungrounded_closes,
        "revised_keys": sorted(state.findings.revised_keys()),
    }


def build_meta(state: AgentRunState, iterations: int) -> AgentLoopMeta:
    closed = len(state.reported_keys)
    return AgentLoopMeta(
        iterations=iterations,
        tool_calls_total=state.tool_calls_total,
        convergence_reason=state.convergence_reason,
        sealed=state.sealed,
        input_tokens_total=state.input_tokens_total(),
        output_tokens_total=sum(ts.output_tokens for ts in state.spend.values()),
        cost_usd_total=sum(ts.cost_usd for ts in state.spend.values()),
        searched_entities=frozenset(state.searched_entities),
        plan_seeded=len(state.plan),
        plan_covered=(
            state.plan_covered_at_stop
            if state.plan_covered_at_stop is not None
            else len(state.plan.keys() & state.addressed)
        ),
        report_calls_total=state.report_calls_total,
        turns_to_first_report=state.turns_to_first_report,
        unknown_aspect_keys=state.unknown_aspect_keys,
        ungrounded_close_rate=(state.ungrounded_closes / closed) if closed else 0.0,
        revised_keys=len(state.findings.revised_keys()),
    )
