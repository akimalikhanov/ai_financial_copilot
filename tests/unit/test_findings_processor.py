"""Unit tests for findings_processor (currency normalization, comparison logic, rendering).

Note: no ISO-4217 validation exists anywhere in this codebase — currencies are
opaque strings end-to-end. This is a possible gap, not something fixed/tested here.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from src.schemas.agent_findings import AgentFindings, Figure, Finding
from src.schemas.retrieval import RAGContext
from src.services.chat.agent.number_grounding import NumberGrounding
from src.services.chat.agent.processor import (
    _normalize_date,
    _render_findings_block,
    process_findings,
    to_millions,
)

FRANKFURTER_BASE = "https://api.frankfurter.dev/v1"
_NO_EXCERPTS = RAGContext(formatted_context="", items=(), chunk_count=0)


class TestNormalizeDate:
    def test_iso_passthrough(self) -> None:
        assert _normalize_date("2023-12-31") == "2023-12-31"

    def test_bare_year_is_not_guessed(self) -> None:
        # Dec 31 is wrong for every non-calendar fiscal year; the latest rate is used and
        # the row says so.
        assert _normalize_date("2023") is None

    def test_invalid_returns_none_and_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING"):
            result = _normalize_date("not-a-date")
        assert result is None
        assert "period_end_not_iso" in caplog.text

    def test_none_input_returns_none(self) -> None:
        assert _normalize_date(None) is None

    def test_empty_string_returns_none(self) -> None:
        assert _normalize_date("") is None


class TestToMillions:
    def test_billion_scales_by_1000(self) -> None:
        assert to_millions(2.0, "B") == 2000.0

    def test_million_scales_by_1(self) -> None:
        assert to_millions(2.0, "M") == 2.0

    def test_thousand_scales_by_0_001(self) -> None:
        assert to_millions(2.0, "K") == pytest.approx(0.002)

    def test_empty_string_scales_by_1e_minus_6(self) -> None:
        assert to_millions(2_000_000.0, "") == pytest.approx(2.0)


def _figure(amount: float, currency: str | None, **overrides: Any) -> Figure:
    fields: dict[str, Any] = {
        "metric": "revenue",
        "amount": amount,
        "unit": "M",
        "currency": currency,
        "period_end": "2023-12-31",
        "fiscal_label": "FY2023",
    }
    fields.update(overrides)
    return Figure(**fields)


def _finding(key: str, *figures: Figure, evidence: list[str] | None = None) -> Finding:
    return Finding(
        key=key,
        claim=f"{key} reported revenue.",
        supported=True,
        evidence=evidence or [],
        confidence="high",
        figures=list(figures),
    )


def _negative(key: str) -> Finding:
    return Finding(
        key=key, claim=f"{key} does not report it.", supported=False, evidence=[], confidence="high"
    )


def _findings(*items: Finding, op: Any = "none") -> AgentFindings:
    return AgentFindings(findings=items, comparison_op=op)


@pytest.fixture(autouse=True)
def _disable_langfuse(monkeypatch: pytest.MonkeyPatch):
    from src.observability import langfuse as lf_client

    monkeypatch.setattr(lf_client, "get_client", lambda: None)


class TestCurrencyResolutionPriority:
    @pytest.mark.asyncio
    async def test_requested_currency_wins(self) -> None:
        result = await process_findings(
            _findings(_finding("A", _figure(100.0, "USD"))), requested_currency="EUR"
        )
        assert result.target_currency == "EUR"

    @pytest.mark.asyncio
    async def test_multi_currency_comparison_defaults_to_usd(self) -> None:
        findings = _findings(
            _finding("A", _figure(100.0, "EUR")), _finding("B", _figure(200.0, "GBP")), op="argmax"
        )
        with respx.mock:
            respx.get(url__startswith=FRANKFURTER_BASE).mock(
                return_value=httpx.Response(200, json={"rates": {"USD": 1.1}})
            )
            result = await process_findings(findings)
        assert result.target_currency == "USD"
        assert result.answer_note is not None
        assert "no target currency specified" in result.answer_note

    @pytest.mark.asyncio
    async def test_no_conversion_needed_same_currency(self) -> None:
        result = await process_findings(_findings(_finding("A", _figure(100.0, "USD"))))
        assert result.target_currency is None
        assert result.currency_converted is False
        assert result.figures[0].normalized_amount == 100.0

    @pytest.mark.asyncio
    async def test_findings_without_figures_pass_through(self) -> None:
        findings = _findings(_finding("A1"), _negative("A2"))
        result = await process_findings(findings)
        assert result.figures == ()
        assert result.findings is findings
        assert result.answer_note is None


class TestPartialFailurePolicy:
    @pytest.mark.asyncio
    async def test_argmax_aborts_entirely_on_any_fx_failure(self) -> None:
        findings = _findings(
            _finding("A", _figure(100.0, "EUR")), _finding("B", _figure(200.0, "USD")), op="argmax"
        )
        with respx.mock:
            respx.get(url__startswith=FRANKFURTER_BASE).mock(return_value=httpx.Response(500))
            result = await process_findings(findings, requested_currency="USD")

        assert result.answer_entity is None
        assert result.currency_converted is False
        assert all(n.normalized_amount is None for n in result.figures)
        assert result.answer_note is not None
        assert "comparison not possible" in result.answer_note

    @pytest.mark.asyncio
    async def test_list_op_degrades_partially_on_fx_failure(self) -> None:
        findings = _findings(
            _finding("A", _figure(100.0, "EUR")), _finding("B", _figure(200.0, "USD")), op="list"
        )
        with respx.mock:
            respx.get(url__startswith=FRANKFURTER_BASE).mock(return_value=httpx.Response(500))
            result = await process_findings(findings, requested_currency="USD")

        assert result.answer_note is not None
        assert "FX conversion failed" in result.answer_note
        by_key = {n.key: n for n in result.figures}
        assert by_key["A"].normalized_amount is None  # failed conversion
        assert by_key["B"].normalized_amount == 200.0  # same currency, no conversion needed


class TestComparisonOp:
    @pytest.mark.asyncio
    async def test_argmin_picks_smallest(self) -> None:
        findings = _findings(
            _finding("A", _figure(300.0, "USD")), _finding("B", _figure(100.0, "USD")), op="argmin"
        )
        assert (await process_findings(findings)).answer_entity == "B"

    @pytest.mark.asyncio
    async def test_argmax_picks_largest(self) -> None:
        findings = _findings(
            _finding("A", _figure(300.0, "USD")), _finding("B", _figure(100.0, "USD")), op="argmax"
        )
        assert (await process_findings(findings)).answer_entity == "A"

    @pytest.mark.asyncio
    async def test_ranking_respects_scale(self) -> None:
        findings = _findings(
            _finding("A", _figure(900.0, "USD", unit="M")),
            _finding("B", _figure(1.0, "USD", unit="B")),
            op="argmax",
        )
        assert (await process_findings(findings)).answer_entity == "B"

    @pytest.mark.asyncio
    async def test_null_currency_excluded_from_ranking(self) -> None:
        findings = _findings(
            _finding("A", _figure(300.0, None)), _finding("B", _figure(100.0, "USD")), op="argmax"
        )
        result = await process_findings(findings)
        assert result.answer_entity == "B"
        assert result.answer_note is not None
        assert "excluded from ranking" in result.answer_note

    @pytest.mark.asyncio
    async def test_unstated_scale_excluded_from_ranking(self) -> None:
        # An unknown scale is never read as millions: 300 of unknown scale could be the
        # smallest or the largest.
        findings = _findings(
            _finding("A", _figure(300.0, "USD", unit=None)),
            _finding("B", _figure(100.0, "USD")),
            op="argmax",
        )
        result = await process_findings(findings)
        assert result.answer_entity == "B"
        assert result.answer_note is not None
        assert "A" in result.answer_note

    @pytest.mark.asyncio
    async def test_several_figures_per_key_are_not_ranked(self) -> None:
        findings = _findings(
            _finding("A", _figure(300.0, "USD"), _figure(250.0, "USD", period_end="2022-12-31")),
            _finding("B", _figure(100.0, "USD")),
            op="argmax",
        )
        result = await process_findings(findings)
        assert result.answer_entity is None
        assert result.answer_note == "not ranked — several figures per entity"

    @pytest.mark.asyncio
    async def test_only_one_available_entity_note(self) -> None:
        findings = _findings(_finding("A", _figure(100.0, "USD")), _negative("B"), op="list")
        result = await process_findings(findings)
        assert result.answer_note == "only one entity had available data"

    @pytest.mark.asyncio
    async def test_one_entity_note_needs_a_comparison(self) -> None:
        # An analytical run with one numeric aspect is not "one entity had data".
        findings = _findings(_finding("A1", _figure(100.0, "USD")), _finding("A2"))
        assert (await process_findings(findings)).answer_note is None


class TestNumberGroundingWiring:
    @pytest.mark.asyncio
    async def test_no_chunk_texts_stays_unverifiable(self) -> None:
        findings = _findings(_finding("A", _figure(100.0, "USD"), evidence=["c1"]))
        result = await process_findings(findings)
        assert result.figures[0].number_grounding is NumberGrounding.UNVERIFIABLE

    @pytest.mark.asyncio
    async def test_missing_chunk_text_is_unverifiable_not_not_found(self) -> None:
        findings = _findings(_finding("A", _figure(100.0, "USD"), evidence=["c1"]))
        result = await process_findings(findings, chunk_texts={})
        assert result.figures[0].number_grounding is NumberGrounding.UNVERIFIABLE

    @pytest.mark.asyncio
    async def test_each_figure_is_checked(self) -> None:
        findings = _findings(
            _finding(
                "A",
                _figure(100.0, "USD"),
                _figure(80.0, "USD", period_end="2022-12-31"),
                evidence=["c1"],
            )
        )
        result = await process_findings(findings, chunk_texts={"c1": "Revenue was 100.0"})
        assert [n.number_grounding for n in result.figures] == [
            NumberGrounding.GROUNDED,
            NumberGrounding.NOT_FOUND,
        ]

    @pytest.mark.asyncio
    async def test_unstated_scale_must_appear_as_printed(self) -> None:
        findings = _findings(_finding("A", _figure(41.2, None, unit=None), evidence=["c1"]))
        result = await process_findings(findings, chunk_texts={"c1": "Gross margin 41.2%"})
        assert result.figures[0].number_grounding is NumberGrounding.GROUNDED

    @pytest.mark.asyncio
    async def test_verifies_native_value_not_fx_converted_value(self) -> None:
        """Regression guard: checking the converted amount would make every converted
        figure read as a false NOT_FOUND, since it never appears in the filing text."""
        findings = _findings(_finding("A", _figure(100.0, "EUR"), evidence=["c1"]))
        with respx.mock:
            respx.get(url__startswith=FRANKFURTER_BASE).mock(
                return_value=httpx.Response(200, json={"rates": {"USD": 1.1}})
            )
            result = await process_findings(
                findings,
                requested_currency="USD",
                chunk_texts={"c1": "Total revenue was EUR 100.0"},
            )
        assert result.figures[0].normalized_amount == pytest.approx(110.0)
        assert result.figures[0].number_grounding is NumberGrounding.GROUNDED


class TestRenderedBlock:
    @pytest.mark.asyncio
    async def test_one_block_for_every_shape(self) -> None:
        findings = AgentFindings(
            findings=(
                _finding(
                    "Acme",
                    _figure(120.0, "USD"),
                    _figure(100.0, "USD", period_end="2022-12-31", fiscal_label="FY2022"),
                ),
                _negative("A2"),
            ),
            conclusion="Revenue grew.",
            unresolved=("Not resolved: Globex",),
        )
        block = _render_findings_block(await process_findings(findings), _NO_EXCERPTS)

        assert block.startswith("[FINDINGS]") and block.endswith("[END FINDINGS]")
        assert "1. Acme [high confidence] Acme reported revenue. | evidence: —" in block
        assert "   - revenue (FY2023 / 2023-12-31): USD 120.0M" in block
        assert "   - revenue (FY2022 / 2022-12-31): USD 100.0M" in block
        # Aspect ids are not rendered — the model would cite them as `[A2]`.
        assert "2. [not disclosed] A2 does not report it." in block
        assert "Conclusion: Revenue grew." in block
        assert "Unresolved: Not resolved: Globex" in block

    @pytest.mark.asyncio
    async def test_unstated_scale_is_said_not_assumed(self) -> None:
        findings = _findings(_finding("A", _figure(41.2, None, unit=None, period_end=None)))
        block = _render_findings_block(await process_findings(findings), _NO_EXCERPTS)
        assert "   - revenue (FY2023): 41.2 (scale not stated)" in block


def _year(amount: float, year: int, **overrides: Any) -> Figure:
    fields: dict[str, Any] = {
        "currency": "USD",
        "period_end": f"{year}-12-31",
        "fiscal_label": f"FY{year}",
    }
    fields.update(overrides)
    return _figure(amount, **fields)


async def _change_lines(
    *figures: Figure, evidence: list[str] | None = None, **kw: Any
) -> list[str]:
    findings = _findings(_finding("Aurora", *figures, evidence=evidence))
    block = _render_findings_block(await process_findings(findings, **kw), _NO_EXCERPTS)
    return [line for line in block.splitlines() if line.startswith("   - change")]


class TestChangeLines:
    @pytest.mark.asyncio
    async def test_up_then_down_from_a_nil_base(self) -> None:
        # Trace c67763e8: nil → 82 → 68 was answered as "a decline over three years".
        lines = await _change_lines(_year(68, 2022), _year(0, 2020), _year(82, 2021))
        assert lines == [
            "   - change 2020-12-31 → 2021-12-31: up USD 82.0M (% change n/m)",
            "   - change 2021-12-31 → 2022-12-31: down USD 14.0M (-17.1%)",
            "   - change 2020-12-31 → 2022-12-31: up USD 68.0M (% change n/m)",
        ]

    @pytest.mark.asyncio
    async def test_mixed_units_compare_in_the_finer_unit(self) -> None:
        lines = await _change_lines(_year(82, 2021), _year(0.068, 2022, unit="B"))
        assert lines == ["   - change 2021-12-31 → 2022-12-31: down USD 14.0M (-17.1%)"]

    @pytest.mark.asyncio
    async def test_unchanged(self) -> None:
        lines = await _change_lines(_year(82, 2021), _year(82, 2022))
        assert lines == ["   - change 2021-12-31 → 2022-12-31: unchanged (+0.0%)"]

    @pytest.mark.asyncio
    async def test_non_monetary_change_has_no_percentage(self) -> None:
        # A margin's relative change would read as a percentage-point change.
        lines = await _change_lines(
            _year(5.5, 2021, currency=None, unit="", metric="operating margin"),
            _year(6.4, 2022, currency=None, unit="", metric="operating margin"),
        )
        assert lines == ["   - change 2021-12-31 → 2022-12-31: up 0.9"]

    @pytest.mark.asyncio
    async def test_metrics_and_currencies_are_kept_apart(self) -> None:
        lines = await _change_lines(
            _year(82, 2021), _year(5, 2022, metric="net loss"), _year(70, 2022, currency="EUR")
        )
        assert lines == []

    @pytest.mark.asyncio
    async def test_shared_period_end_gives_no_change(self) -> None:
        # A 10-Q's three and nine months both end on the quarter's last day.
        lines = await _change_lines(
            _year(30, 2021), _year(40, 2022, fiscal_label="Q3"), _year(90, 2022, fiscal_label="9M")
        )
        assert lines == []

    @pytest.mark.asyncio
    async def test_figures_without_date_or_scale_are_left_out(self) -> None:
        lines = await _change_lines(
            _year(82, 2021), _year(68, 2022, unit=None), _figure(70, "USD", period_end=None)
        )
        assert lines == []

    @pytest.mark.asyncio
    async def test_unverified_figure_marks_its_change(self) -> None:
        lines = await _change_lines(
            _year(82, 2021),
            _year(68, 2022),
            evidence=["c1"],
            chunk_texts={"c1": "(in millions) | Revenue | $ 82 |"},
        )
        assert lines == [
            "   - change 2021-12-31 → 2022-12-31: down USD 14.0M (-17.1%)"
            " | ⚠ UNVERIFIED: computed from an unverified figure"
        ]
