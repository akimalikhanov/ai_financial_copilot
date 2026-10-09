"""Agent state: settings and the run state the loop mutates.

Two kinds of state: `transcript` is the model's view (lossy, compactable); `evidence` + the
record fields below are the durable, never-truncated record. `AgentRunState` is neither
alone — it is transcript + record + loop control.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.services.chat.agent.evidence import EvidenceLedger
from src.services.chat.agent.findings import FindingsLedger
from src.services.chat.agent.tools import (
    REPORT_ANALYTICAL_TOOL,
    REPORT_FINDINGS_TOOL,
    SEARCH_ANALYTICAL_TOOL,
    SEARCH_TOOL,
)
from src.services.chat.agent.transcript import Transcript

if TYPE_CHECKING:
    from src.services.llm_adapters.base_adapter import LLMResponseStats

ConvergenceReason = Literal[
    "natural",
    "convergence",
    "iteration_cap",
    "budget_cap",
    "timeout",
    "covered",  # every planned aspect/entity was reported on
    "search_unavailable",  # every search this turn errored — a dead backend, not an empty corpus
    "deadline",  # wall-clock bound for the whole run
    "llm_error",  # every model in the tool-model chain raised a provider error
    "truncated",  # the tool model hit its completion-token cap without calling a tool
]


class AgentSettings(BaseModel):
    """Validated agent env config. The loop reads these directly; what differs per shape
    is resolved once into a `ShapeConfig`."""

    model_config = ConfigDict(frozen=True)

    tool_model: str
    max_iterations: int = Field(ge=1, le=20)
    # Run spend across every model, priced as billed: output and reasoning at the output
    # rate, cached input at the cached rate. A model with no pricing adds nothing.
    cost_budget_usd: float = Field(gt=0)
    max_concurrent_searches: int = Field(ge=1, le=16)
    max_chunks_per_entity: int = Field(ge=1)
    # Consecutive turns with no new chunks and no key closed before Stop("convergence").
    # At 0 a run stops on the first such turn, before the model has read the grounding
    # feedback from its last report.
    max_empty_rounds: int = Field(ge=0)
    # Wall-clock bound on the whole run. Every tool-model call is budgeted from what is
    # left of it, so a slow call can use the time a fast run would not have needed.
    deadline_seconds: float = Field(gt=0)
    # Ceiling on one tool-model call, so one stalled call cannot use the whole run.
    turn_timeout_cap_seconds: float = Field(gt=0)
    # Run time a tool-model call is never given: its own timeout then fires before the
    # deadline cancels it, and the loop keeps time to act on the failure.
    deadline_reserve_seconds: float = Field(ge=0)
    # Bound on one search (retrieve + rerank).
    search_timeout_seconds: float = Field(gt=0)
    # Analytical runs open several aspects and corroborate each, so they get more turns.
    max_iterations_analytical: int = Field(ge=1, le=20)
    # Ceiling on loop-minted plan entries. Near-duplicate sub_questions each mint their
    # own id (no fuzzy matching), so this is what bounds the cost of that choice.
    max_plan_items: int = Field(ge=1)

    @model_validator(mode="after")
    def _reserve_leaves_run_time(self) -> AgentSettings:
        if self.deadline_reserve_seconds >= self.deadline_seconds:
            raise ValueError(
                "AGENT_DEADLINE_RESERVE_SECONDS must be below AGENT_DEADLINE_SECONDS, "
                "or no tool-model call ever gets a budget"
            )
        return self


@dataclass(frozen=True)
class ShapeConfig:
    """Everything that differs between query shapes. Findings are reported, stored,
    checked and rendered the same way for every shape."""

    prompt: str
    # Analytical searches carry a `sub_question` that mints an aspect key; extraction
    # searches are keyed by the entity they name.
    search_takes_sub_question: bool
    # Analytical findings state their numbers in `claim`; offering `figures` as well gets
    # every number written twice.
    report_takes_figures: bool
    seed_plan_from_entities: bool
    max_iterations: int

    @property
    def tools(self) -> list[dict]:
        search = SEARCH_ANALYTICAL_TOOL if self.search_takes_sub_question else SEARCH_TOOL
        report = REPORT_FINDINGS_TOOL if self.report_takes_figures else REPORT_ANALYTICAL_TOOL
        return [search, report]


def shape_config(query_shape: str | None, settings: AgentSettings) -> ShapeConfig:
    if query_shape == "analytical":
        return ShapeConfig(
            prompt="v6_agent_analytical",
            search_takes_sub_question=True,
            report_takes_figures=False,
            seed_plan_from_entities=False,
            max_iterations=settings.max_iterations_analytical,
        )
    return ShapeConfig(
        prompt="v4_agent",
        search_takes_sub_question=False,
        report_takes_figures=True,
        seed_plan_from_entities=True,
        max_iterations=settings.max_iterations,
    )


def get_agent_settings() -> AgentSettings:
    """Read + validate agent config from env. Not cached — called once per request, so
    env overrides (incl. in tests) always take effect."""
    max_iterations = int(os.getenv("AGENT_MAX_ITERATIONS", "5"))
    return AgentSettings(
        tool_model=os.getenv("AGENT_TOOL_MODEL", "gpt-4o-mini"),
        max_iterations=max_iterations,
        cost_budget_usd=float(os.getenv("AGENT_COST_BUDGET_USD", "0.10")),
        max_concurrent_searches=int(os.getenv("AGENT_MAX_CONCURRENT_SEARCHES", "3")),
        max_chunks_per_entity=int(os.getenv("AGENT_MAX_CHUNKS_PER_ENTITY", "5")),
        max_empty_rounds=int(os.getenv("AGENT_MAX_EMPTY_ROUNDS", "1")),
        deadline_seconds=float(os.getenv("AGENT_DEADLINE_SECONDS", "180")),
        turn_timeout_cap_seconds=float(os.getenv("AGENT_TURN_TIMEOUT_CAP_SECONDS", "120")),
        deadline_reserve_seconds=float(os.getenv("AGENT_DEADLINE_RESERVE_SECONDS", "15")),
        search_timeout_seconds=float(os.getenv("AGENT_SEARCH_TIMEOUT_SECONDS", "60")),
        max_iterations_analytical=int(os.getenv("AGENT_MAX_ITERATIONS_ANALYTICAL", "7")),
        max_plan_items=int(os.getenv("AGENT_MAX_PLAN_ITEMS", "6")),
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
    no chunks and so leaves no ledger trace at all. `errored == searches` means the
    backend was unreachable; `searches > 0` with `new_chunks == 0` means the corpus
    genuinely lacks it.
    """

    searches: int = 0
    errored: int = 0
    new_chunks: int = 0


@dataclass
class AgentRunState:
    """The whole loop state: transcript (view) + record (durable) + control.

    `state.transcript` beside `state.evidence` reads as view-vs-record at every call
    site — that grouping is the point.
    """

    # --- input: not state, but read throughout the loop ---
    settings: AgentSettings
    max_iterations: int  # ShapeConfig.max_iterations

    # --- transcript: the model's view. Append-only, never authoritative. ---
    transcript: Transcript

    # --- durable record: complete, never truncated ---
    evidence: EvidenceLedger = field(default_factory=EvidenceLedger)
    findings: FindingsLedger = field(default_factory=FindingsLedger)  # what we concluded (keyed)
    # Set when Stop("covered") closes every planned key. Read `sealed`, not this — an
    # empty plan can never set it, and would otherwise report "didn't finish covering".
    sealed_by_coverage: bool = False
    # Retrieval capabilities that were unavailable for at least one search this run
    # ("dense", "keyword", "rerank"). A run can be partly degraded without any single
    # search failing outright, so this accumulates rather than describing the last search.
    degraded_capabilities: set[str] = field(default_factory=set)
    # Cleared once any contributing search returned fusion scores instead of rerank ones.
    scores_are_rerank: bool = True
    spend: dict[str, TokenSpend] = field(default_factory=dict)

    # --- plan: entity names seeded by the loop, or aspect ids minted from sub_questions ---
    plan: dict[str, str] = field(default_factory=dict)  # key -> label, insertion-ordered
    aspect_stats: dict[str, AspectStats] = field(default_factory=dict)

    # --- control: loop counters + outcome ---
    iteration: int = 0
    empty_rounds: int = 0
    tool_calls_total: int = 0
    convergence_reason: ConvergenceReason = "iteration_cap"
    # Event-loop time at which the run deadline fires; set when the run starts.
    deadline_at: float | None = None

    # --- instrumentation: is reporting incremental, and how often do calls fail? ---
    report_calls_total: int = 0
    turns_to_first_report: int | None = None
    unknown_aspect_keys: int = 0
    # Stated negatives refused because no earlier turn searched their key.
    unsearched_negatives: int = 0
    # Tool calls whose arguments failed schema validation; each one costs a turn.
    search_arg_errors: int = 0
    report_parse_failures: int = 0
    # Input tokens of the latest tool-model call. The transcript is append-only, so this
    # is how large it grew; it is the signal for adding an overflow valve.
    last_turn_input_tokens: int = 0

    @property
    def addressed(self) -> set[str]:
        """Keys that produced a grounded finding (a stated negative counts).

        Derived from the ledger, never stored, so a key can only close by producing real
        output. Nothing else marks a key done: keys still open at the end of the run are
        rendered as unresolved at projection, which leaves coverage readable at any time.
        """
        return self.findings.keys()

    @property
    def sealed(self) -> bool:
        """Did the run finish what it set out to cover (vs a degraded projection)?

        A plan is empty in two legitimate cases — extraction whose entities resolved to no
        documents, and an analytical run whose searches carried `sub_question: null` — and
        neither means the run fell short. Reading `sealed_by_coverage` alone conflates
        "nothing to cover" with "didn't finish covering", biasing the
        `agent_findings_sealed` metric low and appending a "did not fully converge" caveat
        to answers that were complete.
        """
        return not self.plan or self.sealed_by_coverage

    def record_spend(self, model_id: str, stats: LLMResponseStats | None) -> None:
        if stats is None:
            return
        ts = self.spend.setdefault(model_id, TokenSpend())
        ts.input_tokens += stats.input_tokens or 0
        ts.output_tokens += stats.output_tokens or 0
        ts.cost_usd += stats.cost_usd or 0.0

    def input_tokens_total(self) -> int:
        return sum(ts.input_tokens for ts in self.spend.values())

    def cost_usd_total(self) -> float:
        return sum(ts.cost_usd for ts in self.spend.values())

    def spend_within_budget(self) -> bool:
        return self.cost_usd_total() <= self.settings.cost_budget_usd


@dataclass
class AgentLoopMeta:
    iterations: int
    tool_calls_total: int
    convergence_reason: ConvergenceReason
    # False means the served findings are a degraded projection of accumulated partial
    # findings: the plan was not covered.
    sealed: bool = False
    # Summed across the tool model and its fallbacks; the budget cap checks cost.
    input_tokens_total: int = 0
    output_tokens_total: int = 0
    cost_usd_total: float = 0.0
    # Input tokens of the run's last tool-model call: the size the transcript reached.
    last_turn_input_tokens: int = 0
    # Retrieval capabilities unavailable for at least one search ("dense", "keyword",
    # "rerank"). Drives the user-facing degradation badge and the trace; distinct from a
    # total outage, which surfaces as a search error and gap text instead.
    degraded_capabilities: frozenset[str] = field(default_factory=frozenset)
    # False when the served chunks' scores are fusion-scale, so confidence thresholds
    # calibrated on cross-encoder scores cannot be applied to them.
    scores_are_rerank: bool = True
    # Decomposition width/coverage and whether reporting was incremental.
    plan_seeded: int = 0
    plan_covered: int = 0
    report_calls_total: int = 0
    turns_to_first_report: int | None = None
    unknown_aspect_keys: int = 0
    unsearched_negatives: int = 0
    # Share of positive claims dropped because none of their citations resolved.
    uncited_claim_rate: float = 0.0
    search_arg_errors: int = 0
    report_parse_failures: int = 0
    # Configuration the run actually used, so traces can be sliced by it.
    prompt_version: str | None = None


def open_aspects(state: AgentRunState) -> list[str]:
    """Plan entries that have not yet produced a finding, in mint order."""
    addressed = state.addressed
    return [a for a in state.plan if a not in addressed]


def unresolved_lines(state: AgentRunState) -> list[str]:
    """One stated limitation per open plan key, for the served findings.

    A key whose every search errored was never checked, so it says the backend was down
    rather than "not in the documents". Only a seeded key can be unsearched; a minted one
    exists because a search named it.
    """
    lines: list[str] = []
    for key in open_aspects(state):
        stats = state.aspect_stats.get(key)
        if stats is None or stats.searches == 0:
            lines.append(f"Not searched: {state.plan[key]}")
        elif stats.errored == stats.searches:
            lines.append(
                f"Could not be checked — document search was unavailable: {state.plan[key]}"
            )
        else:
            lines.append(f"Not resolved: {state.plan[key]}")
    return lines


def render_status(state: AgentRunState) -> str | None:
    """Coverage status, computed per call and appended last — never stored.

    Keys and sub-questions only, never claim bodies: the model has nothing to copy, so it
    cannot restate an earlier conclusion instead of reporting a new one. It is the only
    place coverage is stated: nothing stored in the transcript states it, so nothing there
    goes stale. The stall nudge lives here too rather than being appended once and never
    removed.
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
    # Deliberately no "you have evidence but recorded nothing" nudge: turn 1 with nothing
    # recorded is the normal state of a run about to drill down, so such a nudge trades
    # the second search pass for an early report.
    # Final turn: search tools are withheld, so the only move left is converting admitted
    # evidence into findings. Deliberately routes the pressure into `supported: false`
    # rather than into closure — a coerced supported claim citing a real but irrelevant
    # label passes grounding, closes its key, and silently promotes an iteration-capped
    # run to `sealed`, dropping the "did not fully converge" caveat.
    if state.iteration == state.max_iterations - 1:
        parts.append(
            "Final turn — no further searches will run. Report every open key from the "
            "evidence you have already retrieved. If a key is not supported by what you "
            "have, report it with `supported: false` and state the absence in `claim`. "
            "Anything left unreported is recorded as unresolved."
        )
    return " · ".join(parts) if parts else None


def snapshot(state: AgentRunState) -> dict:
    """JSON-safe view of the run's stores, for Langfuse.

    Attached to every `agent_turn_N` span (state after that turn's effects landed) and once
    to the `agent_loop` span when the run ends. Run-level counters are not repeated here:
    they live on `AgentLoopMeta`, which the caller puts on the `agent_loop` span output.
    """
    return {
        "iteration": state.iteration,
        "plan": dict(state.plan),
        "addressed": sorted(state.addressed),
        "open_aspects": open_aspects(state),
        "aspect_stats": {a: vars(s) for a, s in state.aspect_stats.items()},
        "empty_rounds": state.empty_rounds,
        "findings": {
            k: f.model_dump(mode="json")
            for k in sorted(state.findings.keys())
            if (f := state.findings.get(k)) is not None
        },
        "findings_screened": state.findings.screened(),
        "evidence_chunk_count": len(state.evidence),
        "transcript_message_count": len(state.transcript.messages),
    }


def build_meta(
    state: AgentRunState,
    iterations: int,
    *,
    prompt_version: str | None = None,
) -> AgentLoopMeta:
    return AgentLoopMeta(
        iterations=iterations,
        tool_calls_total=state.tool_calls_total,
        convergence_reason=state.convergence_reason,
        sealed=state.sealed,
        input_tokens_total=state.input_tokens_total(),
        output_tokens_total=sum(ts.output_tokens for ts in state.spend.values()),
        cost_usd_total=state.cost_usd_total(),
        degraded_capabilities=frozenset(state.degraded_capabilities),
        scores_are_rerank=state.scores_are_rerank,
        plan_seeded=len(state.plan),
        plan_covered=len(state.plan.keys() & state.addressed),
        report_calls_total=state.report_calls_total,
        turns_to_first_report=state.turns_to_first_report,
        unknown_aspect_keys=state.unknown_aspect_keys,
        unsearched_negatives=state.unsearched_negatives,
        uncited_claim_rate=state.findings.uncited_claim_rate(),
        search_arg_errors=state.search_arg_errors,
        report_parse_failures=state.report_parse_failures,
        last_turn_input_tokens=state.last_turn_input_tokens,
        prompt_version=prompt_version,
    )
