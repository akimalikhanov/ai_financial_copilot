"""Unit tests for AgentSettings and ShapeConfig, per-model spend attribution, and the
loop-minted decomposition plan."""

from __future__ import annotations

from typing import ClassVar
from uuid import uuid4

import pytest
from pydantic import ValidationError

from src.schemas.agent_findings import Finding, FindingsReport
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
    shape_config,
    unresolved_lines,
)
from src.services.chat.agent.transcript import Transcript
from src.services.llm_adapters.base_adapter import LLMResponseStats, ToolCallRef
from tests.unit.test_findings import _f, _neg, _seed_evidence


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


class TestShapeConfig:
    def test_analytical(self) -> None:
        cfg = shape_config("analytical", _settings())
        assert cfg.prompt == "v6_agent_analytical"
        assert cfg.max_iterations == 7
        assert cfg.search_takes_sub_question is True
        assert cfg.seed_plan_from_entities is False

    @pytest.mark.parametrize("shape", ["extraction", "comparison", None])
    def test_other_shapes(self, shape: str | None) -> None:
        cfg = shape_config(shape, _settings())
        assert cfg.prompt == "v4_agent"
        assert cfg.max_iterations == 5
        assert cfg.search_takes_sub_question is False
        assert cfg.seed_plan_from_entities is True

    def test_settings_are_frozen(self) -> None:
        with pytest.raises(ValidationError):
            _settings().max_iterations = 9  # type: ignore[misc]


def _state(**overrides: object) -> AgentRunState:
    settings = overrides.pop("settings", None) or _settings(cost_budget_usd=0.01)
    defaults: dict[str, object] = {
        "settings": settings,
        "max_iterations": settings.max_iterations,  # type: ignore[union-attr]
        "transcript": Transcript([]),
        "evidence": EvidenceLedger(),
    }
    defaults.update(overrides)
    return AgentRunState(**defaults)  # type: ignore[arg-type]


def _report_call(*findings: Finding) -> ToolCallRef:
    return ToolCallRef(
        id="c",
        name="report_findings",
        arguments=FindingsReport(findings=findings).model_dump_json(),
    )


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
        state.findings.record("A1", _f("A1", evidence=[ids[0]]), evidence)
        assert open_aspects(state) == ["A2"]

    def test_empty_plan_has_no_open_aspects(self) -> None:
        assert open_aspects(_state()) == []


class TestUnresolvedLines:
    def test_one_line_per_open_key(self) -> None:
        evidence, ids = _seed_evidence(1)
        state = _state(
            plan={"A1": "q1", "A2": "q2"},
            evidence=evidence,
            aspect_stats={"A2": AspectStats(searches=1)},
        )
        state.findings.record("A1", _f("A1", evidence=[ids[0]]), evidence)
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

    def test_unsearched_seeded_key_reads_as_not_searched(self) -> None:
        """A seeded entity the model never searched is not "not in the documents"."""
        state = _state(
            plan={"Acme": "Acme", "Globex": "Globex"},
            aspect_stats={"Acme": AspectStats(searches=1)},
        )
        assert unresolved_lines(state) == ["Not resolved: Acme", "Not searched: Globex"]

    def test_reading_lines_leaves_coverage_unchanged(self) -> None:
        state = _state(plan={"A1": "q1"})
        unresolved_lines(state)
        assert open_aspects(state) == ["A1"]
        assert state.addressed == set()

    def test_ungrounded_finding_does_not_close_an_aspect_on_its_own(self) -> None:
        """`findings.record` drops an ungrounded item, so a key that only ever produced
        one must stay open rather than closing silently."""
        state = _state(plan={"A1": "q1"})
        ungrounded = _f("A1", evidence=[str(uuid4())])
        assert state.findings.record("A1", ungrounded, state.evidence) is False
        assert open_aspects(state) == ["A1"]
        assert state.addressed == set()

    def test_ungrounded_report_leaves_its_aspect_open_and_unsealed(self) -> None:
        """Closing an ungrounded report's key would reach `Stop("covered")` → `sealed`,
        promoting a run that produced no grounded output for the key to a converged
        run's trust level. Projection renders it as unresolved instead."""
        state = _state(plan={"A1": "Why did gross margin fall?"})

        result = _apply_report(_report_call(_f("A1", evidence=[str(uuid4())])), state, "req-1")

        assert open_aspects(state) == ["A1"]
        assert state.addressed == set()
        assert state.sealed is False
        # The model must be told it did not land, or it never retries the aspect.
        assert "A1 was not recorded" in result
        assert "supported: false" in result

    def test_supported_false_report_closes_its_aspect(self) -> None:
        """The honest exit the final-turn nudge routes toward: a stated negative is
        grounded by definition and closes its key as a real keyed entry."""
        state = _state(
            plan={"A1": "Why did gross margin fall?"}, aspect_stats={"A1": AspectStats(searches=1)}
        )

        result = _apply_report(_report_call(_neg("A1")), state, "req-1")

        assert open_aspects(state) == []
        assert state.addressed == {"A1"}
        assert "Recorded A1." in result


class TestRenderStatusNudges:
    """`render_status` is recomputed per turn and never stored, so what it appends is
    pure per-turn steering — it never accumulates in the transcript and never goes stale."""

    def test_mid_run_status_does_not_push_the_model_to_report(self) -> None:
        """Regression guard. A nudge keyed on "has evidence, recorded nothing" was tried
        and reverted: turn 1 with nothing recorded is the normal state of a run about to
        drill down, so it traded the second search pass for an early report. Mid-run
        turns carry coverage facts only."""
        ledger, _ = _seed_evidence(1)
        state = _state(plan={"A1": "q1"}, evidence=ledger)
        state.iteration = 1

        assert render_status(state) == "Open: A1 (q1)"

    def test_final_turn_routes_pressure_into_supported_false(self) -> None:
        """Deliberately not "close every key": a coerced `supported: true` claim citing a
        real-but-irrelevant label passes grounding, closes its key, and silently promotes
        an iteration-capped run to a converged run's trust level."""
        state = _state(plan={"A1": "q1"})
        state.iteration = state.max_iterations - 1

        status = render_status(state)

        assert status is not None
        assert "Final turn — no further searches will run" in status
        assert "supported: false" in status

    def test_no_final_turn_notice_before_the_last_iteration(self) -> None:
        state = _state(plan={"A1": "q1"})
        state.iteration = state.max_iterations - 2

        status = render_status(state)

        assert status is not None
        assert "Final turn" not in status


class TestSealed:
    def test_empty_plan_is_trivially_sealed(self) -> None:
        """A plan is empty when extraction resolved no documents, or when an analytical
        run's searches carried `sub_question: null`. Neither means the run fell short —
        but `Stop("covered")` requires a plan, so the stored flag can never be set and the
        run would report "didn't finish covering"."""
        assert _state(plan={}).sealed is True

    def test_open_plan_is_not_sealed(self) -> None:
        assert _state(plan={"A1": "q1"}).sealed is False

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
        state.findings.record("A1", _f("A1", evidence=[ids[0]]), evidence)
        projected = state.findings.projection(degraded=not state.sealed)
        assert projected is not None
        assert _DEGRADED_CAVEAT not in projected.unresolved


class TestStatedNegatives:
    """`supported=False` is the typed channel for "searched, not disclosed"."""

    _PLAN: ClassVar[dict[str, str]] = {"A4": "Did FX move the margin?"}

    def test_negative_for_an_unsearched_aspect_is_refused(self) -> None:
        state = _state(plan=self._PLAN)

        result = _apply_report(_report_call(_neg("A4")), state, "req")

        assert open_aspects(state) == ["A4"]
        assert state.unsearched_negatives == 1
        assert "A4 was not recorded as absent — no earlier search covered it." in result

    def test_negative_for_an_unsearched_entity_is_refused(self) -> None:
        state = _state(
            plan={"Acme": "Acme", "Globex": "Globex"},
            aspect_stats={"Acme": AspectStats(searches=1)},
        )

        _apply_report(_report_call(_neg("Acme"), _neg("Globex")), state, "req")

        assert state.addressed == {"Acme"}
        assert state.unsearched_negatives == 1

    def test_negative_closes_its_aspect(self) -> None:
        state = _state(plan=self._PLAN, aspect_stats={"A4": AspectStats(searches=1)})
        _apply_report(_report_call(_neg("A4")), state, "req")
        assert open_aspects(state) == []
        # Not an uncited claim: honesty must not be counted as hallucination, or
        # `uncited_claim_rate` conflates the two.
        assert state.findings.uncited_claim_rate() == 0.0

    def test_later_support_overwrites_the_negative(self) -> None:
        state = _state(plan=self._PLAN, aspect_stats={"A4": AspectStats(searches=1)})
        evidence, ids = _seed_evidence(1)
        state.evidence = evidence
        _apply_report(_report_call(_neg("A4", "No FX disclosure found.")), state, "req")
        _apply_report(
            _report_call(_f("A4", claim="FX was a 3.1pp headwind.", evidence=[ids[0]])),
            state,
            "req",
        )
        projected = state.findings.projection(degraded=False)
        assert projected is not None
        assert [f.claim for f in projected.findings] == ["FX was a 3.1pp headwind."]
        assert projected.unresolved == ()
