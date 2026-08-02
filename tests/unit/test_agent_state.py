"""Unit tests for EffortPrior.for_shape (Stage 1.5's per-query_shape effort prior),
per-model spend attribution, and AgentSettings' validation boundaries (P2-15)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.services.chat.agent.evidence import EvidenceLedger
from src.services.chat.agent.state import (
    AgentRunState,
    AgentSettings,
    EffortPrior,
    get_agent_settings,
)
from src.services.chat.agent.transcript import Transcript
from src.services.llm_adapters.base_adapter import LLMResponseStats


def _settings(**overrides: object) -> AgentSettings:
    defaults: dict[str, object] = {
        "enabled": True,
        "tool_model": "gpt-test",
        "max_iterations": 5,
        "token_budget": 150_000,
        "max_concurrent_searches": 3,
        "max_chunks_per_entity": 5,
        "max_empty_analytical_rounds": 1,
        "max_insufficiency_rejections": 1,
        "max_named_item_rejections_per_item": 2,
        "max_named_item_rejections_total": 10,
        "max_restatement_rejections": 2,
        "turn_timeout_seconds": 60.0,
        "max_iterations_analytical": 7,
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
        assert prior.max_insufficiency_rejections == settings.max_insufficiency_rejections
        assert prior.max_concurrent_searches == settings.max_concurrent_searches

    def test_named_item_caps_are_shape_invariant(self) -> None:
        # The named-item gate is isinstance-scoped to the analytical finalizer (FR-11),
        # so for_shape must carry both caps through unchanged for every shape.
        settings = _settings()
        for shape in ("analytical", "extraction", "comparison", None):
            prior = EffortPrior.for_shape(settings, shape)
            assert (
                prior.max_named_item_rejections_per_item
                == settings.max_named_item_rejections_per_item
            )
            assert prior.max_named_item_rejections_total == settings.max_named_item_rejections_total


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

    def test_named_item_rejection_caps_default_from_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("AGENT_MAX_NAMED_ITEM_REJECTIONS_PER_ITEM", raising=False)
        monkeypatch.delenv("AGENT_MAX_NAMED_ITEM_REJECTIONS_TOTAL", raising=False)
        settings = get_agent_settings()
        assert settings.max_named_item_rejections_per_item == 2
        assert settings.max_named_item_rejections_total == 10

    def test_named_item_total_cap_of_zero_is_the_kill_switch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # ge=0, not ge=1: an operator must be able to disable the mechanism entirely
        # via env without a code change (plan §6 rollback lever 1).
        monkeypatch.setenv("AGENT_MAX_NAMED_ITEM_REJECTIONS_TOTAL", "0")
        assert get_agent_settings().max_named_item_rejections_total == 0
        assert _settings(max_named_item_rejections_total=0).max_named_item_rejections_total == 0
        assert (
            _settings(max_named_item_rejections_per_item=0).max_named_item_rejections_per_item == 0
        )

    def test_named_item_caps_reject_negative(self) -> None:
        with pytest.raises(ValidationError):
            _settings(max_named_item_rejections_total=-1)
        with pytest.raises(ValidationError):
            _settings(max_named_item_rejections_per_item=-1)
