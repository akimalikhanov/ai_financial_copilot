"""Unit tests for the synchronous parts of one turn: `fold_searches`, `fold_reports`,
`decide`. No LLM, no event loop — state in, state and outcome out."""

from __future__ import annotations

import json

from src.schemas.agent_findings import Observation
from src.services.chat.agent.loop import (
    Continue,
    Stop,
    TurnFacts,
    _SearchResult,
    decide,
    fold_reports,
    fold_searches,
)
from src.services.llm_adapters.base_adapter import ToolCallRef
from tests.unit.test_agent_loop_smoke import _make_chunk_with_payload
from tests.unit.test_agent_state import _settings, _state


def _search(call_id: str) -> ToolCallRef:
    return ToolCallRef(id=call_id, name="search_documents", arguments="{}")


def _report(call_id: str, aspect: str, chunk_id: str) -> ToolCallRef:
    return ToolCallRef(
        id=call_id,
        name="report_analytical_findings",
        arguments=json.dumps(
            {
                "question": "q",
                "observations": [
                    {
                        "aspect": aspect,
                        "claim": "c",
                        "evidence_chunks": [chunk_id],
                        "confidence": "high",
                    }
                ],
            }
        ),
    )


def _facts(**overrides: object) -> TurnFacts:
    defaults: dict[str, object] = {
        "searches": 1,
        "backend_failures": 0,
        "new_chunks": 1,
        "closed": frozenset(),
    }
    defaults.update(overrides)
    return TurnFacts(**defaults)  # type: ignore[arg-type]


class TestFoldSearches:
    def test_labels_continue_across_searches_and_echo_the_aspect(self) -> None:
        state = _state(plan={"A1": "Did costs rise?"})
        c1, p1 = _make_chunk_with_payload()
        c2, p2 = _make_chunk_with_payload()
        results = [
            _SearchResult(entity="Acme", chunks=[c1], payloads=p1),
            _SearchResult(entity="Acme", chunks=[c2], payloads=p2),
        ]

        texts, new = fold_searches(
            state, [_search("s1"), _search("s2")], results, {"s1": "A1", "s2": None}, "rw"
        )

        assert new == [1, 1]
        assert texts["s1"].startswith("[A1] ") and 'id="S1"' in texts["s1"]
        assert 'id="S2"' in texts["s2"]
        assert state.searched_entities == {"Acme"}
        assert vars(state.aspect_stats["A1"]) == {"searches": 1, "errored": 0, "new_chunks": 1}

    def test_failed_search_returns_its_error_and_counts_against_the_aspect(self) -> None:
        state = _state(plan={"A1": "q"})
        failed = _SearchResult(
            entity="Acme", chunks=[], payloads={}, error_str="down", backend_failed=True
        )

        texts, new = fold_searches(state, [_search("s1")], [failed], {"s1": "A1"}, "rw")

        assert texts == {"s1": "down"}
        assert new == [0]
        assert state.aspect_stats["A1"].errored == 1


class TestFoldReports:
    def test_returns_the_keys_this_turn_closed(self) -> None:
        state = _state(plan={"A1": "q", "A2": "r"})
        chunk, payloads = _make_chunk_with_payload()
        state.evidence.admit([chunk])
        state.evidence.assign_labels([chunk], payloads)

        texts, closed = fold_reports(state, [_report("r1", "A1", "S1")], "req")

        assert closed == frozenset({"A1"})
        assert texts["r1"].startswith("Recorded A1.")

    def test_an_ungrounded_report_closes_nothing(self) -> None:
        state = _state(plan={"A1": "q"})

        _texts, closed = fold_reports(state, [_report("r1", "A1", "S9")], "req")

        assert closed == frozenset()


class TestDecide:
    def test_covered_plan_seals(self) -> None:
        state = _state(plan={"A1": "q"})
        negative = Observation(
            aspect="A1", claim="c", substantiated=False, evidence_chunks=[], confidence="high"
        )
        state.findings.record("A1", negative, state.evidence)

        assert decide(state, _facts()) == Stop("covered")
        assert state.sealed_by_coverage

    def test_every_search_failing_stops(self) -> None:
        state = _state(plan={"A1": "q"})
        outcome = decide(state, _facts(searches=2, backend_failures=2, new_chunks=0))
        assert outcome == Stop("search_unavailable")

    def test_empty_rounds_stop_only_past_the_tolerance(self) -> None:
        state = _state(plan={"A1": "q"}, settings=_settings(max_empty_rounds=1))
        empty = _facts(new_chunks=0)

        assert decide(state, empty) == Continue()
        assert decide(state, empty) == Stop("convergence")

    def test_progress_resets_the_empty_round_count(self) -> None:
        state = _state(plan={"A1": "q", "A2": "r"}, empty_rounds=1)
        assert decide(state, _facts(new_chunks=0, closed=frozenset({"A1"}))) == Continue()
        assert state.empty_rounds == 0
