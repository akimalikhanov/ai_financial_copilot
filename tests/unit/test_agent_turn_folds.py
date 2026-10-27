"""Unit tests for the synchronous parts of one turn: `fold_searches`, `fold_reports`,
`decide`. No LLM, no event loop — state in, state and outcome out."""

from __future__ import annotations

import json
from dataclasses import replace as dc_replace
from typing import Any
from uuid import uuid4

import pytest

from src.schemas.agent_findings import Finding
from src.services.chat.activity import (
    NoteRow,
    SearchSources,
    search_result_row,
    search_sources,
)
from src.services.chat.agent.findings import FLAGGED_MARKER
from src.services.chat.agent.loop import (
    Continue,
    Stop,
    TurnFacts,
    _SearchResult,
    decide,
    fold_reports,
    fold_searches,
)
from src.services.chat.agent.state import AgentRunState, AspectStats
from src.services.llm_adapters.base_adapter import (
    AssistantTurnResult,
    LLMResponseStats,
    ToolCallRef,
)
from tests.unit.test_agent_loop_smoke import (
    _capture_activity,
    _make_chunk_with_payload,
    _make_state,
    _run,
)
from tests.unit.test_agent_state import _report_call, _settings, _state
from tests.unit.test_findings import _BLOCKED, _FLAGGED, _f, _neg, _seed_evidence


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

        texts, new, traces, _ = fold_searches(
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

        texts, _, traces, _ = fold_searches(
            state, [_search("s1")], [_SearchResult("Acme", [c], p)], {"s1": "A1"}
        )

        [hit] = traces["s1"]["chunks"]  # type: ignore[index]
        assert len(hit["text"]) == 10 and hit["text"] in texts["s1"]

    def test_failed_search_returns_its_error_and_counts_against_the_aspect(self) -> None:
        state = _state(plan={"A1": "q"})
        failed = _SearchResult(
            entity="Acme", chunks=[], payloads={}, error_str="down", backend_failed=True
        )

        texts, new, traces, _ = fold_searches(state, [_search("s1")], [failed], {"s1": "A1"})

        assert texts == traces == {"s1": "down"}
        assert new == [0]
        assert state.aspect_stats["A1"].errored == 1

    def test_not_found_search_closes_its_entity_key_as_a_negative(self) -> None:
        state = _state(plan={"Aurora": "Aurora", "RWE AG": "RWE AG"})
        missing = _SearchResult(
            entity="Aurora", chunks=[], payloads={}, error_str="no match", not_found=True
        )

        texts, _, _, _ = fold_searches(state, [_search("s1")], [missing], {"s1": "Aurora"})

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

        texts, labels, _, _ = fold_searches(
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

        texts, labels, _, _ = fold_searches(
            state, [_search("s2")], [_SearchResult("Acme", [old], p_old)], {"s2": "A1"}
        )

        assert labels == [0]
        assert texts["s2"] == "[A1] Already shown above: S1"

    def test_a_chunk_below_the_render_cut_is_no_progress(self) -> None:
        state = _state(plan={"A1": "q"}, settings=_settings(max_chunks_per_entity=1))
        top, p_top = _make_chunk_with_payload()
        below, p_below = _make_chunk_with_payload()
        fold_searches(state, [_search("s1")], [_SearchResult("Acme", [top], p_top)], {"s1": "A1"})

        _texts, labels, _, _ = fold_searches(
            state,
            [_search("s2")],
            [_SearchResult("Acme", [top, below], p_top | p_below)],
            {"s2": "A1"},
        )

        assert labels == [0]  # `below` was admitted but never shown
        assert below.chunk_id in state.evidence

    def test_an_empty_search_says_no_results(self) -> None:
        state = _state(plan={"A1": "q"})
        texts, _, _, _ = fold_searches(
            state, [_search("s1")], [_SearchResult("Acme", [], {})], {"s1": "A1"}
        )
        assert texts["s1"] == "[A1] (no results)"

    def test_sources_group_shown_pages_by_document(self) -> None:
        state = _state(plan={"A1": "q"})
        c1, p1 = _make_chunk_with_payload()
        c2, p2 = _make_chunk_with_payload()
        p2 = {c2.chunk_id: dc_replace(p2[c2.chunk_id], page_numbers=(42, 1))}

        *_, sources = fold_searches(
            state, [_search("s1")], [_SearchResult("Acme", [c1, c2], p1 | p2)], {"s1": "A1"}
        )

        assert sources == [[("Acme 10-K", [1, 42])]]


def _row(new: int = 0, **result: Any) -> tuple[str, str]:
    chunks = result.pop("chunks", [])
    return search_result_row(
        _SearchResult(entity="Acme", chunks=chunks, payloads={}, **result), new, timeout_s=60
    )


class TestSearchResultRow:
    def test_new_excerpts(self) -> None:
        chunk, _ = _make_chunk_with_payload()
        assert _row(8, chunks=[chunk]) == ("ok", "+8 new")

    def test_sources_list_up_to_max(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENT_ACTIVITY_MAX_SOURCES", "2")
        sources: SearchSources = [("RBA 10-K 2023", [42, 88]), ("RBA 10-K 2022", [49]), ("X", [])]
        assert search_sources(sources) == "RBA 10-K 2023 p.42, p.88 · RBA 10-K 2022 p.49"
        assert search_sources([]) is None

    def test_chunks_returned_none_new(self) -> None:
        chunk, _ = _make_chunk_with_payload()
        assert _row(0, chunks=[chunk]) == ("skipped", "nothing new")

    def test_no_chunks(self) -> None:
        assert _row(0) == ("skipped", "no results")

    def test_degraded(self) -> None:
        chunk, _ = _make_chunk_with_payload()
        assert _row(8, chunks=[chunk], degraded=frozenset({"rerank"})) == (
            "warn",
            "+8 new · reranking unavailable",
        )

    def test_not_in_library(self) -> None:
        assert _row(not_found=True, error_str="x") == (
            "skipped",
            "Acme is not in your documents",
        )

    def test_timed_out(self) -> None:
        assert _row(timed_out=True, backend_failed=True, error_str="x") == (
            "warn",
            "timed out after 60s",
        )

    def test_backend_failed(self) -> None:
        assert _row(backend_failed=True, error_str="x") == ("warn", "search failed")

    def test_invalid_arguments(self) -> None:
        assert _row(args_invalid=True, error_str="x") == ("skipped", "malformed search, skipped")


class TestFoldReports:
    def test_returns_the_keys_this_turn_closed(self) -> None:
        state = _state(plan={"A1": "q", "A2": "r"})
        chunk, payloads = _make_chunk_with_payload()
        state.evidence.admit([chunk])
        state.evidence.assign_labels([chunk], payloads)

        texts, closed, _ = fold_reports(state, [_report("r1", "A1", "S1")], "req")

        assert closed == frozenset({"A1"})
        assert texts["r1"].startswith("Recorded A1.")

    def test_an_ungrounded_report_closes_nothing(self) -> None:
        state = _state(plan={"A1": "q"})

        _texts, closed, _ = fold_reports(state, [_report("r1", "A1", "S9")], "req")

        assert closed == frozenset()


def _note_rows(state: AgentRunState, *findings: Finding) -> list[NoteRow]:
    _, _, rows = fold_reports(state, [_report_call(*findings)], "req")
    return rows


class TestNoteRows:
    def test_kept_supported_finding(self) -> None:
        state = _state(plan={"A1": "Revenue 2023?"})
        state.evidence, ids = _seed_evidence(1)
        claim = "Revenue was <n>$1.73B</n>."
        rows = _note_rows(state, _f("A1", claim=claim, evidence=[ids[0]]))
        assert rows == [NoteRow("ok", "Noted: Revenue 2023?", claim="Revenue was $1.73B.")]

    def test_kept_negative(self) -> None:
        state = _state(plan={"A1": "FX impact?"}, aspect_stats={"A1": AspectStats(searches=1)})
        rows = _note_rows(state, _neg("A1", "No FX impact is disclosed."))
        assert rows == [
            NoteRow("skipped", "Not in documents: FX impact?", claim="No FX impact is disclosed.")
        ]

    def test_flagged_claim_shows_cleaned_text(self) -> None:
        state = _state(plan={"A1": "q"})
        state.evidence, ids = _seed_evidence(1)
        [row] = _note_rows(state, _f("A1", claim=_FLAGGED, evidence=[ids[0]]))
        stored = state.findings.get("A1")
        assert stored is not None
        assert row == NoteRow(
            "ok", "Noted: q", "text cleaned", stored.claim.removeprefix(FLAGGED_MARKER)
        )
        assert row.claim is not None and FLAGGED_MARKER not in row.claim

    def test_unknown_key(self) -> None:
        state = _state(plan={"A1": "q"})
        assert _note_rows(state, _neg("A9")) == [
            NoteRow("warn", "Note not recorded — wrong key, retried")
        ]

    def test_negative_without_a_prior_search(self) -> None:
        state = _state(plan={"A1": "q"})
        assert _note_rows(state, _neg("A1")) == [
            NoteRow("warn", "Note not recorded — not searched yet")
        ]

    def test_claim_citing_no_retrieved_excerpt(self) -> None:
        state = _state(plan={"A1": "q"})
        assert _note_rows(state, _f("A1", evidence=[str(uuid4())])) == [
            NoteRow("warn", "Note not recorded — no supporting excerpt")
        ]

    def test_blocked_claim(self) -> None:
        state = _state(plan={"A1": "q"})
        state.evidence, ids = _seed_evidence(1)
        assert _note_rows(state, _f("A1", claim=_BLOCKED, evidence=[ids[0]])) == [
            NoteRow("warn", "Note not recorded — blocked by safety check")
        ]

    def test_arguments_did_not_parse(self) -> None:
        state = _state(plan={"A1": "q"})
        malformed = ToolCallRef(id="r1", name="report_findings", arguments="{")
        _, _, rows = fold_reports(state, [malformed], "req")
        assert rows == [NoteRow("warn", "Note not recorded — malformed")]

    def test_rows_follow_report_order_across_rejections(self) -> None:
        state = _state(plan={"A1": "q", "A2": "r"}, aspect_stats={"A2": AspectStats(searches=1)})
        state.evidence, ids = _seed_evidence(1)
        rows = _note_rows(
            state, _neg("A9"), _f("A1", claim="x", evidence=[ids[0]]), _neg("A2", "y")
        )
        assert [r.label for r in rows] == [
            "Note not recorded — wrong key, retried",
            "Noted: q",
            "Not in documents: r",
        ]


@pytest.mark.asyncio
async def test_blocked_claim_text_reaches_no_event(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_COST_BUDGET_USD", "100")
    events = _capture_activity(monkeypatch)
    state = _make_state()
    chunk, payloads = _make_chunk_with_payload()
    search = ToolCallRef(
        id="s1", name="search_documents", arguments=json.dumps({"entity": "Acme", "query": "q"})
    )
    blocked = _f("Acme", claim=_BLOCKED, evidence=["S1"])
    turns = [
        AssistantTurnResult(text="", tool_calls=[search]),
        AssistantTurnResult(text="", tool_calls=[_report_call(blocked)]),
        AssistantTurnResult(text="done", tool_calls=[]),
    ]

    async def _search(*_a: Any, **_k: Any) -> _SearchResult:
        return _SearchResult(entity="Acme", chunks=[chunk], payloads=payloads)

    await _run(state, turns, _search)

    notes = [e for e in events if e["kind"] == "note"]
    assert [n["label"] for n in notes] == ["Note not recorded — blocked by safety check"]
    assert _BLOCKED not in json.dumps(events)


class TestDecide:
    def test_covered_plan_seals(self) -> None:
        state = _state(plan={"A1": "q"})
        state.findings.record("A1", _neg("A1"), state.evidence)

        assert decide(state, _facts(new_labels=0)) == Stop("covered")
        assert state.sealed_by_coverage

    def test_covered_plan_with_unread_results_continues(self) -> None:
        # The turn reported the last key and searched too; the model has not read those
        # results yet, so stopping would discard a follow-up it paid for.
        state = _state(plan={"A1": "q"})
        state.findings.record("A1", _neg("A1"), state.evidence)

        assert decide(state, _facts(new_labels=2)) == Continue()
        assert not state.sealed_by_coverage

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
