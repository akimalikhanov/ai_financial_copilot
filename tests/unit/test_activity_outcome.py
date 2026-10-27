"""`build_outcome`: one case per scenario in the timeline plan (A1–E7)."""

from __future__ import annotations

from typing import Any

import pytest

from src.services.chat.activity import Outcome, OutcomeInputs, build_outcome
from tests.unit.test_activity_stages import _RBA, _companies, _docs, _meta

_COVERED = "Stopped: everything planned was covered"
_RBA_SCOPE = _companies(_RBA, years=(2022, 2023))
_RBA_2023 = _companies(_RBA, years=(2023,))


def _agent(scope: Any = _RBA_SCOPE, shape: str = "extraction", **kw: Any) -> OutcomeInputs:
    meta_kw = {k: kw.pop(k) for k in list(kw) if k not in OutcomeInputs.__dataclass_fields__}
    return OutcomeInputs("agent", scope=scope, query_shape=shape, meta=_meta(**meta_kw), **kw)


def _stopped(reason: str, text: str) -> Any:
    return pytest.param(
        _agent(convergence_reason=reason, unresolved=("Not resolved: Microsoft Corporation",)),
        Outcome(
            "warn",
            f"Stopped early: {text}",
            [f"Stopped: {text}", "Not resolved: Microsoft Corporation"],
        ),
        id=f"D_{reason}",
    )


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        pytest.param(
            _agent(),
            Outcome("ok", f"Searched {_RBA} (2023, 2022) · 7 searches", [_COVERED]),
            id="A1_extraction_covered",
        ),
        pytest.param(
            _agent(report_calls_total=3, tool_calls_total=10, iterations=4),
            Outcome("ok", f"Searched {_RBA} (2023, 2022) · 7 searches", [_COVERED]),
            id="A2_rejected_report_retried",
        ),
        pytest.param(
            _agent(
                _companies("Apple Inc.", "Microsoft Corporation", "SAP SE"),
                "comparison",
                tool_calls_total=8,
            ),
            Outcome("ok", "Searched 3 companies · 6 searches", [_COVERED]),
            id="B1_comparison_all_found",
        ),
        pytest.param(
            _agent(
                _companies("Apple Inc.", "Microsoft Corporation", missing=("SAP SE",)),
                "comparison",
            ),
            Outcome("warn", "Searched 2 of 3 companies — SAP SE not in your documents", [_COVERED]),
            id="B2_company_missing",
        ),
        pytest.param(
            _agent(shape="analytical", tool_calls_total=11),
            Outcome("ok", f"Searched {_RBA} (2023, 2022) · 9 searches", [_COVERED]),
            id="C1_analytical_covered",
        ),
        pytest.param(
            _agent(_docs("all", 14), "analytical", tool_calls_total=8),
            Outcome("ok", "Searched all 14 documents · 6 searches", [_COVERED]),
            id="C2_whole_library",
        ),
        pytest.param(
            _agent(
                shape="analytical",
                convergence_reason="iteration_cap",
                sealed=False,
                unresolved=(
                    "Not resolved: Did currency movements affect margins?",
                    "Not searched: How did inventory write-downs change?",
                ),
            ),
            Outcome(
                "warn",
                "Stopped early: reached the search limit",
                [
                    "Stopped: reached the search limit",
                    "Not resolved: Did currency movements affect margins?",
                    "Not searched: How did inventory write-downs change?",
                ],
            ),
            id="C3_stopped_with_items_open",
        ),
        pytest.param(
            _agent(tool_calls_total=10),
            Outcome("ok", f"Searched {_RBA} (2023, 2022) · 8 searches", [_COVERED]),
            id="D1_one_search_timed_out_run_sealed",
        ),
        pytest.param(
            _agent(_RBA_2023, degraded_capabilities=frozenset({"rerank"})),
            Outcome(
                "warn",
                f"Searched {_RBA} (2023) · partial search (reranking unavailable)",
                [_COVERED, "Some searches ran without reranking; results may be less relevant."],
            ),
            id="D2_degraded",
        ),
        pytest.param(
            _agent(_RBA_2023, convergence_reason="search_unavailable", sealed=False),
            Outcome(
                "fail",
                "Document search unavailable — answer not grounded",
                ["Stopped: document search was unavailable"],
            ),
            id="D3_search_down",
        ),
        _stopped("deadline", "ran out of time"),
        _stopped("budget_cap", "reached the cost limit"),
        _stopped("timeout", "the search model took too long"),
        _stopped("llm_error", "the search model failed"),
        _stopped("truncated", "the search model's reply was cut off"),
        pytest.param(
            _agent(
                _RBA_2023,
                convergence_reason="natural",
                sealed=False,
                synthesis_fallback="no_findings",
            ),
            Outcome(
                "warn",
                f"Searched {_RBA} (2023) · no findings recorded, answered from excerpts",
                [
                    "Stopped: the search model stopped before covering everything",
                    "No findings were recorded; the answer is drawn directly from retrieved "
                    "excerpts.",
                ],
            ),
            id="D7_answered_from_excerpts",
        ),
        pytest.param(OutcomeInputs("error"), Outcome("fail", "Something went wrong"), id="D8"),
        pytest.param(
            OutcomeInputs("direct", carried_outcome="none"),
            Outcome("ok", "Answered without searching documents"),
            id="E1_direct",
        ),
        pytest.param(
            OutcomeInputs("direct", carried_outcome="carried"),
            Outcome("ok", "Reused findings from your previous question"),
            id="E2_carried",
        ),
        pytest.param(
            OutcomeInputs("direct", carried_outcome="dropped_scope"),
            Outcome(
                "warn",
                "Earlier findings don't match the current scope — answered without documents",
            ),
            id="E2_dropped_scope",
        ),
        pytest.param(
            OutcomeInputs("direct", carried_outcome="dropped_hop_cap"),
            Outcome("warn", "Earlier findings too far back to reuse — answered without documents"),
            id="E2_dropped_hop_cap",
        ),
        pytest.param(
            OutcomeInputs("clarification"), Outcome("ok", "", hidden=True), id="E3_clarification"
        ),
        pytest.param(
            OutcomeInputs("too_broad", too_broad_count=14, max_companies=5),
            Outcome("skipped", "Not run: the question covers 14 companies; up to 5 per question"),
            id="E4_too_broad",
        ),
        pytest.param(
            OutcomeInputs("out_of_scope"),
            Outcome("skipped", "Not answered: outside what this assistant covers"),
            id="E5_out_of_scope",
        ),
        pytest.param(
            OutcomeInputs("blocked"),
            Outcome("fail", "Request blocked by the safety check"),
            id="E6_blocked",
        ),
        pytest.param(
            _agent(),
            Outcome("ok", f"Searched {_RBA} (2023, 2022) · 7 searches", [_COVERED]),
            id="E7_flagged_input_follows_the_run",
        ),
    ],
)
def test_outcome(inputs: OutcomeInputs, expected: Outcome) -> None:
    assert build_outcome(inputs) == expected


def test_natural_and_sealed_reads_finished() -> None:
    outcome = build_outcome(_agent(convergence_reason="natural"))
    assert outcome.status == "ok" and outcome.lines == ["Stopped: finished"]


def test_missing_company_on_an_analytical_run_warns_in_the_lines() -> None:
    outcome = build_outcome(_agent(_companies(_RBA, missing=("SAP SE",)), "analytical"))
    assert outcome.status == "warn"
    assert outcome.lines == [_COVERED, "Not in your documents: SAP SE"]


def test_to_detail() -> None:
    assert Outcome("warn", "t", ["a"]).to_detail() == {
        "status": "warn",
        "text": "t",
        "lines": ["a"],
        "hidden": False,
    }
