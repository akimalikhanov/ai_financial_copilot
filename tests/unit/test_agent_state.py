"""Unit tests for EffortPrior.for_shape (Stage 1.5's per-query_shape effort prior),
per-model spend attribution, AgentSettings' validation boundaries (P2-15), and the
loop-minted decomposition plan (10b step 1)."""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from src.schemas.agent_findings import Observation
from src.services.chat.agent.evidence import EvidenceLedger
from src.services.chat.agent.loop import _mint as _mint_with_cap
from src.services.chat.agent.state import (
    AgentRunState,
    AgentSettings,
    EffortPrior,
    open_aspects,
)
from src.services.chat.agent.transcript import Transcript
from src.services.llm_adapters.base_adapter import LLMResponseStats
from tests.unit.test_findings import _seed_evidence


def _settings(**overrides: object) -> AgentSettings:
    defaults: dict[str, object] = {
        "enabled": True,
        "tool_model": "gpt-test",
        "max_iterations": 5,
        "token_budget": 150_000,
        "max_concurrent_searches": 3,
        "max_chunks_per_entity": 5,
        "max_empty_analytical_rounds": 1,
        "turn_timeout_seconds": 60.0,
        "deadline_seconds": 180.0,
        "max_iterations_analytical": 7,
        "history_turns": 2,
        "history_assistant_tokens": 600,
        "max_plan_items": 8,
        "max_revivals_per_turn": 3,
    }
    defaults.update(overrides)
    return AgentSettings(**defaults)  # type: ignore[arg-type]


class TestEffortPriorForShape:
    def test_analytical_uses_its_own_iteration_cap(self) -> None:
        prior = EffortPrior.for_shape(_settings(), "analytical")
        assert prior.max_iterations == 7

    def test_non_analytical_shapes_use_max_iterations(self) -> None:
        for shape in ("extraction", "comparison", None):
            prior = EffortPrior.for_shape(_settings(), shape)
            assert prior.max_iterations == 5

    def test_defaults_to_shape_invariant_when_not_tuned(self) -> None:
        # get_agent_settings() defaults max_iterations_analytical to max_iterations
        # when AGENT_MAX_ITERATIONS_ANALYTICAL is unset — for_shape must then be a
        # no-op across shapes.
        settings = _settings(max_iterations=5, max_iterations_analytical=5)
        assert EffortPrior.for_shape(settings, "analytical").max_iterations == 5
        assert EffortPrior.for_shape(settings, "extraction").max_iterations == 5

    def test_carries_shape_independent_fields_unchanged(self) -> None:
        settings = _settings()
        prior = EffortPrior.for_shape(settings, "analytical")
        assert prior.max_empty_rounds == settings.max_empty_analytical_rounds
        assert prior.max_concurrent_searches == settings.max_concurrent_searches
        # Every per-turn bound the loop enforces reaches it through the effort prior —
        # no module-level constant beside the config it would silently disagree with.
        assert prior.max_plan_items == settings.max_plan_items
        assert prior.max_revivals_per_turn == settings.max_revivals_per_turn
        assert prior.max_chunks_per_lookup == settings.max_chunks_per_entity


def _state(**overrides: object) -> AgentRunState:
    defaults: dict[str, object] = {
        "effort": EffortPrior.for_shape(_settings(), None),
        "token_budget": 1000,
        "turn_timeout_seconds": 60.0,
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
        state = _state(token_budget=100)
        state.record_spend("model-a", LLMResponseStats(input_tokens=100))
        assert state.spend_within_budget() is True

    def test_over_budget_is_not_within(self) -> None:
        state = _state(token_budget=100)
        state.record_spend("model-a", LLMResponseStats(input_tokens=101))
        assert state.spend_within_budget() is False


class TestAgentSettingsValidation:
    def test_max_iterations_below_minimum_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _settings(max_iterations=0)

    def test_max_iterations_above_maximum_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _settings(max_iterations=21)

    def test_token_budget_below_minimum_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _settings(token_budget=999)

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
    def test_shrinks_as_keys_are_addressed(self) -> None:
        """D4: `addressed` is derived from real output, so an aspect can only close by
        producing a grounded finding or a stated gap — never by being asserted closed."""
        state = _state(plan={"A1": "q1", "A2": "q2", "A3": "q3"})
        assert open_aspects(state) == ["A1", "A2", "A3"]

        state.findings.add_gap("Not resolved: q2", closes="A2")
        assert open_aspects(state) == ["A1", "A3"]

        evidence, ids = _seed_evidence(1)
        state.evidence = evidence
        state.findings.record(
            "A1",
            Observation(aspect="A1", claim="c", evidence_chunks=[ids[0]], confidence="high"),
            evidence,
        )
        state.findings.add_gap("Not resolved: q3", closes="A3")
        assert open_aspects(state) == []

    def test_empty_plan_has_no_open_aspects(self) -> None:
        assert open_aspects(_state()) == []

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
