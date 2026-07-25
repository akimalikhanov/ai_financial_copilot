"""Unit tests for finalizer gates (docs/stages/agentic_state_refactor_v2.md step 6)."""

from __future__ import annotations

from src.schemas.agent_findings import AgentFindings, AnalyticalFindings, EntityFinding, Observation
from src.services.chat.agent import gates as gates_module
from src.services.chat.agent.evidence import EvidenceLedger
from src.services.chat.agent.findings import FindingsLedger
from src.services.chat.agent.state import AgentRunState, EffortPrior
from src.services.chat.agent.transcript import Transcript


def _reported(*entities: str) -> FindingsLedger:
    """A FindingsLedger keyed on the given entities (available=false is grounding-exempt,
    so no EvidenceLedger seeding is needed to give each a key)."""
    ledger = FindingsLedger()
    for name in entities:
        ledger.record(name, EntityFinding(entity=name, available=False), EvidenceLedger())
    return ledger


def _state(**overrides: object) -> AgentRunState:
    defaults: dict[str, object] = {
        "effort": EffortPrior(
            max_iterations=4,
            max_empty_rounds=1,
            max_insufficiency_rejections=1,
            max_concurrent_searches=3,
        ),
        "token_budget": 1_000_000,
        "turn_timeout_seconds": 60.0,
        "transcript": Transcript([]),
        "evidence": EvidenceLedger(),
        "expected_entities": set(),
    }
    defaults.update(overrides)
    return AgentRunState(**defaults)  # type: ignore[arg-type]


class TestMissingEntityGate:
    def test_fires_when_entity_never_searched(self) -> None:
        state = _state(
            expected_entities={"Acme", "Globex"},
            searched_entities={"Acme"},
            findings=_reported("Acme"),
        )
        candidate = AgentFindings(metric_requested="revenue", findings=())
        reason = gates_module.missing_entity_gate(candidate, state)
        assert reason is not None
        assert "search" in reason and "Globex" in reason

    def test_fires_when_searched_but_unreported(self) -> None:
        """The gap this change closes: an entity searched but omitted from the report
        used to slip past both this gate and synthesis' unsearched-stub backstop."""
        state = _state(
            expected_entities={"Acme"}, searched_entities={"Acme"}, findings=FindingsLedger()
        )
        candidate = AgentFindings(metric_requested="revenue", findings=())
        reason = gates_module.missing_entity_gate(candidate, state)
        assert reason is not None
        assert "report" in reason and "Acme" in reason

    def test_fires_when_reported_but_unsearched(self) -> None:
        """The dual hole stays closed: reporting available=false without searching does
        not wave an entity through — the must-retrieve guarantee is preserved."""
        state = _state(
            expected_entities={"Acme"}, searched_entities=set(), findings=_reported("Acme")
        )
        candidate = AgentFindings(metric_requested="revenue", findings=())
        reason = gates_module.missing_entity_gate(candidate, state)
        assert reason is not None
        assert "search" in reason and "Acme" in reason

    def test_silent_when_all_searched_and_reported(self) -> None:
        state = _state(
            expected_entities={"Acme"}, searched_entities={"Acme"}, findings=_reported("Acme")
        )
        candidate = AgentFindings(metric_requested="revenue", findings=())
        assert gates_module.missing_entity_gate(candidate, state) is None

    def test_ignores_analytical_candidates(self) -> None:
        state = _state(expected_entities={"Acme"}, searched_entities=set())
        candidate = AnalyticalFindings(question="q", observations=())
        assert gates_module.missing_entity_gate(candidate, state) is None

    def test_fires_regardless_of_retry_budget(self) -> None:
        """Unlike the sufficiency gate, this never waives on iteration/budget pressure."""
        state = _state(
            expected_entities={"Acme"},
            searched_entities=set(),
            iteration=3,  # last possible iteration
        )
        candidate = AgentFindings(metric_requested="revenue", findings=())
        assert gates_module.missing_entity_gate(candidate, state) is not None


class TestAnalyticalInsufficiencyGate:
    def _thin_candidate(self) -> AnalyticalFindings:
        return AnalyticalFindings(
            question="q",
            observations=(
                Observation(aspect="a", claim="x", evidence_chunks=[], confidence="high"),
            ),
        )

    def test_fires_when_retry_budget_remains(self) -> None:
        state = _state(iteration=0)
        reason = gates_module.analytical_insufficiency_gate(self._thin_candidate(), state)
        assert reason is not None

    def test_waives_on_last_iteration(self) -> None:
        state = _state(iteration=3)  # effort.max_iterations - 1
        assert gates_module.analytical_insufficiency_gate(self._thin_candidate(), state) is None

    def test_waives_when_rejection_cap_reached(self) -> None:
        state = _state(iteration=0, insufficiency_rejections=1)  # == max_insufficiency_rejections
        assert gates_module.analytical_insufficiency_gate(self._thin_candidate(), state) is None

    def test_ignores_extraction_candidates(self) -> None:
        state = _state(iteration=0)
        candidate = AgentFindings(metric_requested="revenue", findings=())
        assert gates_module.analytical_insufficiency_gate(candidate, state) is None


class TestContractC3:
    def test_structural_reason_wins_over_sufficiency(self) -> None:
        """A candidate that fails a coverage gate *and* a sufficiency gate gets the
        coverage reason — gates are ordered structural before sufficiency and the
        dispatcher stops at the first firing reason."""
        from src.services.chat.agent.tools import gates_for

        state = _state(expected_entities={"Acme"}, searched_entities=set(), iteration=0)
        # An AgentFindings candidate only ever runs the missing_entity_gate (the only
        # gate registered for report_findings) — confirm it's first/only and fires.
        registered = gates_for("report_findings")
        assert registered[0] is gates_module.missing_entity_gate

        candidate = AgentFindings(metric_requested="revenue", findings=())
        reason = next((r for g in registered if (r := g(candidate, state)) is not None), None)
        assert reason is not None and "Acme" in reason
