"""Unit tests for AgentSettings (per-shape iteration cap, validation boundaries),
per-model spend attribution, and the loop-minted decomposition plan."""

from __future__ import annotations

from typing import ClassVar
from uuid import uuid4

import pytest
from pydantic import ValidationError

from src.schemas.agent_findings import (
    AgentFindings,
    AnalyticalFindings,
    EntityFinding,
    Observation,
)
from src.services.chat.agent.evidence import EvidenceLedger
from src.services.chat.agent.findings import _DEGRADED_CAVEAT
from src.services.chat.agent.loop import _apply_report
from src.services.chat.agent.loop import _mint as _mint_with_cap
from src.services.chat.agent.state import (
    AgentRunState,
    AgentSettings,
    AspectStats,
    open_aspects,
    render_status,
    unresolved_lines,
)
from src.services.chat.agent.transcript import Transcript
from src.services.llm_adapters.base_adapter import LLMResponseStats, ToolCallRef
from tests.unit.test_findings import _seed_evidence


def _settings(**overrides: object) -> AgentSettings:
    defaults: dict[str, object] = {
        "tool_model": "gpt-test",
        "max_iterations": 5,
        "cost_budget_usd": 0.10,
        "max_concurrent_searches": 3,
        "max_chunks_per_entity": 5,
        "max_empty_rounds": 1,
        "turn_timeout_seconds": 60.0,
        "deadline_seconds": 180.0,
        "max_iterations_analytical": 7,
        "max_plan_items": 8,
    }
    defaults.update(overrides)
    return AgentSettings(**defaults)  # type: ignore[arg-type]


class TestMaxIterationsFor:
    def test_analytical_uses_its_own_iteration_cap(self) -> None:
        assert _settings().max_iterations_for("analytical") == 7

    def test_non_analytical_shapes_use_max_iterations(self) -> None:
        for shape in ("extraction", "comparison", None):
            assert _settings().max_iterations_for(shape) == 5

    def test_settings_are_frozen(self) -> None:
        with pytest.raises(ValidationError):
            _settings().max_iterations = 9  # type: ignore[misc]


def _state(**overrides: object) -> AgentRunState:
    settings = overrides.pop("settings", None) or _settings(cost_budget_usd=0.01)
    defaults: dict[str, object] = {
        "settings": settings,
        "max_iterations": settings.max_iterations_for(None),  # type: ignore[union-attr]
        "transcript": Transcript([]),
        "evidence": EvidenceLedger(),
        "expected_entities": set(),
    }
    defaults.update(overrides)
    return AgentRunState(**defaults)  # type: ignore[arg-type]


class TestSpendAttribution:
    def test_record_spend_accumulates_per_model(self) -> None:
        state = _state()
        state.record_spend("model-a", LLMResponseStats(input_tokens=10, output_tokens=1))
        state.record_spend("model-a", LLMResponseStats(input_tokens=5, output_tokens=2))
        state.record_spend("model-b", LLMResponseStats(input_tokens=100, output_tokens=0))

        assert state.spend["model-a"].input_tokens == 15
        assert state.spend["model-a"].output_tokens == 3
        assert state.spend["model-b"].input_tokens == 100

    def test_input_tokens_total_sums_across_models(self) -> None:
        state = _state()
        state.record_spend("model-a", LLMResponseStats(input_tokens=10))
        state.record_spend("model-b", LLMResponseStats(input_tokens=100))
        assert state.input_tokens_total() == 110

    def test_record_spend_ignores_missing_stats(self) -> None:
        state = _state()
        state.record_spend("model-a", None)
        assert state.spend == {}


class TestSpendWithinBudget:
    def test_at_budget_is_within(self) -> None:
        state = _state()
        state.record_spend("model-a", LLMResponseStats(cost_usd=0.006))
        state.record_spend("model-b", LLMResponseStats(cost_usd=0.004))
        assert state.spend_within_budget() is True

    def test_over_budget_is_not_within(self) -> None:
        state = _state()
        state.record_spend("model-a", LLMResponseStats(cost_usd=0.006))
        state.record_spend("model-b", LLMResponseStats(cost_usd=0.0041))
        assert state.spend_within_budget() is False

    def test_cheap_input_does_not_hide_expensive_output(self) -> None:
        # 8k input and 4k reasoning on gpt-5-mini: the output is 80% of the bill.
        state = _state()
        state.record_spend(
            "model-a", LLMResponseStats(input_tokens=8_000, output_tokens=4_000, cost_usd=0.01)
        )
        state.record_spend("model-a", LLMResponseStats(input_tokens=100, cost_usd=0.001))
        assert state.spend_within_budget() is False


class TestAgentSettingsValidation:
    def test_max_iterations_below_minimum_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _settings(max_iterations=0)

    def test_max_iterations_above_maximum_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _settings(max_iterations=21)

    def test_cost_budget_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            _settings(cost_budget_usd=0)

    def test_turn_timeout_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            _settings(turn_timeout_seconds=0)


_MAX_PLAN_ITEMS = 8  # AGENT_MAX_PLAN_ITEMS default; these tests are about minting, not config


def _mint(plan: dict[str, str], sub_question: str | None) -> str | None:
    return _mint_with_cap(plan, sub_question, _MAX_PLAN_ITEMS)


class TestPlanMinting:
    def test_seeds_plan_and_mints_sequential_ids(self) -> None:
        plan: dict[str, str] = {}
        assert _mint(plan, "Did input costs rise?") == "A1"
        assert _mint(plan, "Did pricing offset them?") == "A2"
        assert plan == {"A1": "Did input costs rise?", "A2": "Did pricing offset them?"}

    def test_identical_sub_question_reuses_id(self) -> None:
        plan: dict[str, str] = {}
        first = _mint(plan, "Did input costs rise?")
        assert _mint(plan, "Did input costs rise?") == first
        assert len(plan) == 1

    def test_case_and_whitespace_variants_reuse_id(self) -> None:
        plan: dict[str, str] = {}
        first = _mint(plan, "Did input costs rise?")
        assert _mint(plan, "  did INPUT   costs rise? ") == first
        assert len(plan) == 1

    def test_near_duplicate_mints_a_new_id(self) -> None:
        """No fuzzy matching — the cost is one redundant plan entry, bounded by the cap."""
        plan: dict[str, str] = {}
        assert _mint(plan, "Did input costs rise?") == "A1"
        assert _mint(plan, "Did input cost rise?") == "A2"

    def test_blank_sub_question_returns_none(self) -> None:
        plan: dict[str, str] = {}
        assert _mint(plan, None) is None
        assert _mint(plan, "   ") is None
        assert plan == {}

    def test_cap_holds_and_returns_none_without_raising(self) -> None:
        plan: dict[str, str] = {}
        for i in range(_MAX_PLAN_ITEMS):
            assert _mint(plan, f"question {i}") == f"A{i + 1}"
        assert _mint(plan, "one too many") is None
        assert len(plan) == _MAX_PLAN_ITEMS
        # An already-tracked aspect is still resolvable past the cap.
        assert _mint(plan, "question 0") == "A1"


class TestOpenAspects:
    def test_shrinks_only_as_findings_land(self) -> None:
        """`addressed` is derived from the ledger, so an aspect can only close by
        producing a grounded finding — never by being asserted closed."""
        state = _state(plan={"A1": "q1", "A2": "q2"})
        assert open_aspects(state) == ["A1", "A2"]

        evidence, ids = _seed_evidence(1)
        state.evidence = evidence
        state.findings.record(
            "A1",
            Observation(aspect="A1", claim="c", evidence_chunks=[ids[0]], confidence="high"),
            evidence,
        )
        assert open_aspects(state) == ["A2"]

    def test_empty_plan_has_no_open_aspects(self) -> None:
        assert open_aspects(_state()) == []


class TestUnresolvedLines:
    def test_one_line_per_open_key(self) -> None:
        evidence, ids = _seed_evidence(1)
        state = _state(plan={"A1": "q1", "A2": "q2"}, evidence=evidence)
        state.findings.record(
            "A1",
            Observation(aspect="A1", claim="c", evidence_chunks=[ids[0]], confidence="high"),
            evidence,
        )
        assert unresolved_lines(state) == ["Not resolved: q2"]

    def test_all_errored_searches_read_as_unchecked(self) -> None:
        """A key whose every search errored was never checked; saying "not resolved"
        would read as "not in the documents"."""
        state = _state(
            plan={"A1": "q1", "A2": "q2"},
            aspect_stats={
                "A1": AspectStats(searches=2, errored=2),
                "A2": AspectStats(searches=2, errored=1),
            },
        )
        assert unresolved_lines(state) == [
            "Could not be checked — document search was unavailable: q1",
            "Not resolved: q2",
        ]

    def test_reading_lines_leaves_coverage_unchanged(self) -> None:
        state = _state(plan={"A1": "q1"})
        unresolved_lines(state)
        assert open_aspects(state) == ["A1"]
        assert state.addressed == set()

    def test_ungrounded_finding_does_not_close_an_aspect_on_its_own(self) -> None:
        """The D4 hazard: `findings.record` drops an ungrounded item (C6), so a key that
        only ever produced one must stay open rather than closing silently."""
        state = _state(plan={"A1": "q1"})
        ungrounded = Observation(
            aspect="A1", claim="c", evidence_chunks=[str(uuid4())], confidence="high"
        )
        assert state.findings.record("A1", ungrounded, state.evidence) is False
        assert open_aspects(state) == ["A1"]
        assert state.addressed == set()

    def test_ungrounded_report_leaves_its_aspect_open_and_unsealed(self) -> None:
        """No mid-loop gap reconciliation: gapping an ungrounded report here would close
        its key → `addressed` → `Stop("covered")` → `sealed`, promoting a run that
        produced no grounded output for the aspect to a converged run's trust level.
        Projection renders it as unresolved instead."""
        state = _state(plan={"A1": "Why did gross margin fall?"})
        tc = ToolCallRef(
            id="c1",
            name="report_analytical_findings",
            arguments=AnalyticalFindings(
                question="q",
                observations=(
                    Observation(
                        aspect="A1",
                        claim="c",
                        evidence_chunks=[str(uuid4())],
                        confidence="high",
                    ),
                ),
            ).model_dump_json(),
        )

        result = _apply_report(tc, state, "req-1")

        assert state.findings._gaps is None
        assert open_aspects(state) == ["A1"]
        assert state.addressed == set()
        assert state.sealed is False
        # The model must be told it did not land, or it never retries the aspect.
        assert "A1 was not recorded" in result
        assert "substantiated: false" in result

    def test_substantiated_false_report_closes_its_aspect(self) -> None:
        """The honest exit the final-turn nudge routes toward: a stated negative is
        grounded by definition and closes its key as a real keyed entry."""
        state = _state(
            plan={"A1": "Why did gross margin fall?"}, aspect_stats={"A1": AspectStats(searches=1)}
        )
        tc = ToolCallRef(
            id="c1",
            name="report_analytical_findings",
            arguments=AnalyticalFindings(
                question="q",
                observations=(
                    Observation(
                        aspect="A1",
                        claim="The filings do not disclose a driver for this.",
                        evidence_chunks=[],
                        substantiated=False,
                        confidence="low",
                    ),
                ),
            ).model_dump_json(),
        )

        result = _apply_report(tc, state, "req-1")

        assert open_aspects(state) == []
        assert state.addressed == {"A1"}
        assert "Recorded A1." in result


class TestRenderStatusNudges:
    """`render_status` is recomputed per turn and never stored, so what it appends is
    pure per-turn steering — it never accumulates in the transcript and never goes stale."""

    def test_mid_run_status_does_not_push_the_model_to_report(self) -> None:
        """Regression guard. A nudge keyed on "has evidence, recorded nothing" was tried
        and reverted: turn 1 with nothing recorded is the normal state of a run about to
        drill down, so it traded the second search pass for an early report and turned an
        aspect the follow-up search *did* answer into `substantiated: false` (traces
        af7363d9 vs df7654f6). Mid-run turns carry coverage facts only."""
        ledger, _ = _seed_evidence(1)
        state = _state(plan={"A1": "q1"}, evidence=ledger)
        state.iteration = 1

        status = render_status(state)

        assert status == "Open: A1 (q1)"

    def test_final_turn_routes_pressure_into_substantiated_false(self) -> None:
        """Deliberately not "close every aspect": a coerced `substantiated: true` claim
        citing a real-but-irrelevant label passes grounding, closes its key, and silently
        promotes an iteration-capped run to a converged run's trust level."""
        state = _state(plan={"A1": "q1"})
        state.iteration = state.max_iterations - 1

        status = render_status(state)

        assert status is not None
        assert "Final turn — no further searches will run" in status
        assert "substantiated: false" in status
        assert "must close every aspect" not in status

    def test_no_final_turn_notice_before_the_last_iteration(self) -> None:
        state = _state(plan={"A1": "q1"})
        state.iteration = state.max_iterations - 2

        status = render_status(state)

        assert status is not None
        assert "Final turn" not in status


class TestSealed:
    def test_empty_plan_is_trivially_sealed(self) -> None:
        """P2-I: a plan is empty when extraction resolved no documents, or when an
        analytical run's searches carried `sub_question: null`. Neither means the run fell
        short — but `Stop("covered")` requires a plan, so the stored flag can never be set
        and the run would report "didn't finish covering"."""
        assert _state(plan={}).sealed is True

    def test_open_plan_is_not_sealed(self) -> None:
        state = _state(plan={"A1": "q1"})
        assert state.sealed is False

    def test_covered_plan_is_sealed(self) -> None:
        state = _state(plan={"A1": "q1"})
        state.sealed_by_coverage = True
        assert state.sealed is True

    def test_empty_plan_serves_findings_undegraded(self) -> None:
        """The user-visible half: `projection(degraded=not sealed)` would otherwise append
        "the search did not fully converge" to a complete answer."""
        state = _state(plan={})
        evidence, ids = _seed_evidence(1)
        state.evidence = evidence
        state.findings.record(
            "A1",
            Observation(aspect="A1", claim="c", evidence_chunks=[ids[0]], confidence="high"),
            evidence,
        )
        projected = state.findings.projection(analytical=True, degraded=not state.sealed)
        assert isinstance(projected, AnalyticalFindings)
        assert _DEGRADED_CAVEAT not in (projected.gaps or [])


class TestStatedNegatives:
    """`Observation.substantiated=False` is the analytical counterpart to
    `EntityFinding(available=False)` — the typed channel for "searched, not disclosed"."""

    _PLAN: ClassVar[dict[str, str]] = {"A4": "Did FX move the margin?"}
    _NEGATIVE = Observation(
        aspect="A4",
        claim="The filings do not quantify FX impact.",
        substantiated=False,
        evidence_chunks=[],
        confidence="high",
    )

    def _report(self, state, *observations) -> str:
        tc = ToolCallRef(
            id="c",
            name="report_analytical_findings",
            arguments=AnalyticalFindings(question="q", observations=observations).model_dump_json(),
        )
        return _apply_report(tc, state, "req")

    def test_negative_for_an_unsearched_aspect_is_refused(self) -> None:
        state = _state(plan=self._PLAN)

        result = self._report(state, self._NEGATIVE)

        assert open_aspects(state) == ["A4"]
        assert state.unsearched_negatives == 1
        assert "A4 was not recorded as absent — no earlier search covered it." in result

    def test_negative_for_an_unsearched_entity_is_refused(self) -> None:
        state = _state(plan={"Acme": "Acme", "Globex": "Globex"}, searched_entities={"Acme"})
        tc = ToolCallRef(
            id="c",
            name="report_findings",
            arguments=AgentFindings(
                metric_requested="revenue",
                findings=(
                    EntityFinding(entity="Acme", available=False, reason="not disclosed"),
                    EntityFinding(entity="Globex", available=False, reason="not disclosed"),
                ),
            ).model_dump_json(),
        )

        _apply_report(tc, state, "req")

        assert state.addressed == {"Acme"}
        assert state.unsearched_negatives == 1

    def test_negative_closes_its_aspect_without_a_gap(self) -> None:
        """Before the typed channel, the only way to state a negative was a `gaps` string,
        which closed nothing — the loop kept searching an aspect the model had settled."""
        state = _state(plan=self._PLAN, aspect_stats={"A4": AspectStats(searches=1)})
        self._report(
            state,
            Observation(
                aspect="A4",
                claim="The filings do not quantify FX impact.",
                substantiated=False,
                evidence_chunks=[],
                confidence="high",
            ),
        )
        assert open_aspects(state) == []
        assert state.findings._gaps is None
        # Not an uncited claim: honesty must not be counted as hallucination, or
        # `uncited_claim_rate` conflates the two.
        assert state.findings.uncited_claim_rate() == 0.0

    def test_later_substantiation_overwrites_the_negative(self) -> None:
        """P1-B dissolved: per-aspect state lives in the keyed entry store, so a negative
        superseded by real evidence is overwritten — there is no gap left to retract."""
        state = _state(plan=self._PLAN, aspect_stats={"A4": AspectStats(searches=1)})
        evidence, ids = _seed_evidence(1)
        state.evidence = evidence
        self._report(
            state,
            Observation(
                aspect="A4",
                claim="No FX disclosure found.",
                substantiated=False,
                evidence_chunks=[],
                confidence="high",
            ),
        )
        self._report(
            state,
            Observation(
                aspect="A4",
                claim="FX was a 3.1pp headwind.",
                substantiated=True,
                evidence_chunks=[ids[0]],
                confidence="high",
            ),
        )
        projected = state.findings.projection(analytical=True, degraded=False)
        assert isinstance(projected, AnalyticalFindings)
        assert [o.claim for o in projected.observations] == ["FX was a 3.1pp headwind."]
        assert not projected.gaps
