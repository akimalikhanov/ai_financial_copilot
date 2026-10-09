"""Unit tests for the synchronous parts of one turn: `fold_searches`, `fold_reports`,
`decide`. No LLM, no event loop — state in, state and outcome out."""

from __future__ import annotations

import json
from dataclasses import replace as dc_replace

import pytest

from src.services.chat.agent.loop import (
    Continue,
    Stop,
    TurnFacts,
    _SearchResult,
    decide,
    fold_reports,
    fold_searches,
)
from src.services.llm_adapters.base_adapter import LLMResponseStats, ToolCallRef
from tests.unit.test_agent_loop_smoke import _make_chunk_with_payload
from tests.unit.test_agent_state import _settings, _state
from tests.unit.test_findings import _neg


def _search(call_id: str) -> ToolCallRef:
    return ToolCallRef(id=call_id, name="search_documents", arguments="{}")


def _report(call_id: str, key: str, chunk_id: str) -> ToolCallRef:
    return ToolCallRef(
        id=call_id,
        name="report_findings",
        arguments=json.dumps(
            {
                "findings": [
                    {
                        "key": key,
                        "claim": "c",
                        "supported": True,
                        "evidence": [chunk_id],
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
        "new_labels": 1,
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

        texts, new, traces = fold_searches(
            state, [_search("s1"), _search("s2")], results, {"s1": "A1", "s2": None}
        )

        assert new == [1, 1]
        assert texts["s1"].startswith("[A1] ") and 'id="S1"' in texts["s1"]
        assert 'id="S2"' in texts["s2"]
        assert set(state.aspect_stats) == {"A1"}
        assert vars(state.aspect_stats["A1"]) == {"searches": 1, "errored": 0, "new_chunks": 1}
        assert traces["s1"]["chunks"][0]["ref"] == "S1"  # type: ignore[index]
        assert traces["s1"]["chunks"][0]["chunk_id"] == str(c1.chunk_id)  # type: ignore[index]

    def test_trace_view_trims_chunk_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENT_TRACE_CHUNK_CHARS", "10")
        state = _state(plan={"A1": "q"})
        c, p = _make_chunk_with_payload()

        texts, _, traces = fold_searches(
            state, [_search("s1")], [_SearchResult("Acme", [c], p)], {"s1": "A1"}
        )

        [hit] = traces["s1"]["chunks"]  # type: ignore[index]
        assert len(hit["text"]) == 10 and hit["text"] in texts["s1"]

    def test_failed_search_returns_its_error_and_counts_against_the_aspect(self) -> None:
        state = _state(plan={"A1": "q"})
        failed = _SearchResult(
            entity="Acme", chunks=[], payloads={}, error_str="down", backend_failed=True
        )

        texts, new, traces = fold_searches(state, [_search("s1")], [failed], {"s1": "A1"})

        assert texts == traces == {"s1": "down"}
        assert new == [0]
        assert state.aspect_stats["A1"].errored == 1

    def test_not_found_search_closes_its_entity_key_as_a_negative(self) -> None:
        state = _state(plan={"Aurora": "Aurora", "RWE AG": "RWE AG"})
        missing = _SearchResult(
            entity="Aurora", chunks=[], payloads={}, error_str="no match", not_found=True
        )

        texts, _, _ = fold_searches(state, [_search("s1")], [missing], {"s1": "Aurora"})

        assert texts["s1"] == "no match"
        finding = state.findings.get("Aurora")
        assert finding is not None and finding.supported is False
        assert decide(state, _facts(new_labels=0, closed=frozenset({"Aurora"}))) == Continue()
        state.findings.record("RWE AG", _neg("RWE AG"), state.evidence)
        assert decide(state, _facts(new_labels=0)) == Stop("covered")

    def test_a_re_returned_chunk_is_named_not_rendered_again(self) -> None:
        state = _state(plan={"A1": "q"})
        old, p_old = _make_chunk_with_payload()
        old = dc_replace(old, heading_trail=["Consolidated Statements of Operations"])
        new, p_new = _make_chunk_with_payload()
        fold_searches(state, [_search("s1")], [_SearchResult("Acme", [old], p_old)], {"s1": "A1"})

        texts, labels, _ = fold_searches(
            state,
            [_search("s2")],
            [_SearchResult("Acme", [old, new], p_old | p_new)],
            {"s2": "A1"},
        )

        assert labels == [1]
        assert 'id="S2"' in texts["s2"] and 'id="S1"' not in texts["s2"]
        assert texts["s2"].endswith("Already shown above: S1 Consolidated Statements of Operations")

    def test_a_search_returning_only_shown_chunks_is_no_progress(self) -> None:
        """Not "(no results)": the search found chunks, and saying otherwise sends the
        model searching again for text already on screen."""
        state = _state(plan={"A1": "q"})
        old, p_old = _make_chunk_with_payload()
        fold_searches(state, [_search("s1")], [_SearchResult("Acme", [old], p_old)], {"s1": "A1"})

        texts, labels, _ = fold_searches(
            state, [_search("s2")], [_SearchResult("Acme", [old], p_old)], {"s2": "A1"}
        )

        assert labels == [0]
        assert texts["s2"] == "[A1] Already shown above: S1"

    def test_a_chunk_below_the_render_cut_is_no_progress(self) -> None:
        state = _state(plan={"A1": "q"}, settings=_settings(max_chunks_per_entity=1))
        top, p_top = _make_chunk_with_payload()
        below, p_below = _make_chunk_with_payload()
        fold_searches(state, [_search("s1")], [_SearchResult("Acme", [top], p_top)], {"s1": "A1"})

        _texts, labels, _ = fold_searches(
            state,
            [_search("s2")],
            [_SearchResult("Acme", [top, below], p_top | p_below)],
            {"s2": "A1"},
        )

        assert labels == [0]  # `below` was admitted but never shown
        assert below.chunk_id in state.evidence

    def test_an_empty_search_says_no_results(self) -> None:
        state = _state(plan={"A1": "q"})
        texts, _, _ = fold_searches(
            state, [_search("s1")], [_SearchResult("Acme", [], {})], {"s1": "A1"}
        )
        assert texts["s1"] == "[A1] (no results)"


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
        state.findings.record("A1", _neg("A1"), state.evidence)

        assert decide(state, _facts()) == Stop("covered")
        assert state.sealed_by_coverage

    def test_every_search_failing_stops(self) -> None:
        state = _state(plan={"A1": "q"})
        outcome = decide(state, _facts(searches=2, backend_failures=2, new_labels=0))
        assert outcome == Stop("search_unavailable")

    def test_empty_rounds_stop_only_past_the_tolerance(self) -> None:
        state = _state(plan={"A1": "q"}, settings=_settings(max_empty_rounds=1))
        empty = _facts(new_labels=0)

        assert decide(state, empty) == Continue()
        assert decide(state, empty) == Stop("convergence")

    def test_progress_resets_the_empty_round_count(self) -> None:
        state = _state(plan={"A1": "q", "A2": "r"}, empty_rounds=1)
        assert decide(state, _facts(new_labels=0, closed=frozenset({"A1"}))) == Continue()
        assert state.empty_rounds == 0

    def test_spend_over_the_usd_budget_stops(self) -> None:
        state = _state(plan={"A1": "q"}, settings=_settings(cost_budget_usd=0.01))
        state.record_spend("tool", LLMResponseStats(output_tokens=6_000, cost_usd=0.012))
        assert decide(state, _facts()) == Stop("budget_cap")
