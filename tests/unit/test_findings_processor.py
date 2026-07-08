"""Unit tests for findings_processor (currency normalization, comparison logic).

Note: no ISO-4217 validation exists anywhere in this codebase — currencies are
opaque strings end-to-end. This is a possible gap, not something fixed/tested here.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from src.schemas.agent_findings import AgentFindings, AnalyticalFindings, EntityFinding, Observation
from src.services.chat.findings_processor import _normalize_date, _to_millions, process_findings

FRANKFURTER_BASE = "https://api.frankfurter.dev/v1"


class TestNormalizeDate:
    def test_iso_passthrough(self) -> None:
        assert _normalize_date("2023-12-31") == "2023-12-31"

    def test_bare_year_becomes_dec_31(self) -> None:
        assert _normalize_date("2023") == "2023-12-31"

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
        assert _to_millions(2.0, "B") == 2000.0

    def test_million_scales_by_1(self) -> None:
        assert _to_millions(2.0, "M") == 2.0

    def test_thousand_scales_by_0_001(self) -> None:
        assert _to_millions(2.0, "K") == pytest.approx(0.002)

    def test_empty_string_scales_by_1e_minus_6(self) -> None:
        assert _to_millions(2_000_000.0, "") == pytest.approx(2.0)

    def test_none_unit_scales_by_1(self) -> None:
        assert _to_millions(2.0, None) == 2.0


def _finding(entity: str, value: float | None, currency: str | None, **overrides) -> EntityFinding:
    defaults = {
        "entity": entity,
        "available": value is not None,
        "value": value,
        "currency": currency,
        "period_end": "2023-12-31",
        "unit": "M",
    }
    defaults.update(overrides)
    return EntityFinding(**defaults)


@pytest.fixture(autouse=True)
def _disable_langfuse(monkeypatch: pytest.MonkeyPatch):
    from src.observability import langfuse as lf_client

    monkeypatch.setattr(lf_client, "get_client", lambda: None)


class TestProcessFindingsAnalyticalPassthrough:
    @pytest.mark.asyncio
    async def test_analytical_findings_short_circuits(self) -> None:
        analytical = AnalyticalFindings(
            question="why?",
            observations=(Observation(claim="x", evidence_chunks=[], confidence="high"),),
        )
        result = await process_findings(analytical)
        assert result.findings == ()
        assert result.answer_entity is None
        assert result.analytical_findings is analytical


class TestCurrencyResolutionPriority:
    @pytest.mark.asyncio
    async def test_requested_currency_wins(self) -> None:
        findings = AgentFindings(
            metric_requested="revenue",
            findings=(_finding("A", 100.0, "USD"),),
            comparison_op="none",
        )
        result = await process_findings(findings, requested_currency="EUR")
        assert result.target_currency == "EUR"

    @pytest.mark.asyncio
    async def test_multi_currency_comparison_defaults_to_usd(self) -> None:
        findings = AgentFindings(
            metric_requested="revenue",
            findings=(_finding("A", 100.0, "EUR"), _finding("B", 200.0, "GBP")),
            comparison_op="argmax",
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
        findings = AgentFindings(
            metric_requested="revenue",
            findings=(_finding("A", 100.0, "USD"),),
            comparison_op="none",
        )
        result = await process_findings(findings)
        assert result.target_currency is None
        assert result.currency_converted is False
        assert result.findings[0].normalized_value == 100.0


class TestPartialFailurePolicy:
    @pytest.mark.asyncio
    async def test_argmax_aborts_entirely_on_any_fx_failure(self) -> None:
        findings = AgentFindings(
            metric_requested="revenue",
            findings=(_finding("A", 100.0, "EUR"), _finding("B", 200.0, "USD")),
            comparison_op="argmax",
        )
        with respx.mock:
            respx.get(url__startswith=FRANKFURTER_BASE).mock(return_value=httpx.Response(500))
            result = await process_findings(findings, requested_currency="USD")

        assert result.answer_entity is None
        assert result.currency_converted is False
        assert all(nf.normalized_value is None for nf in result.findings)
        assert result.answer_note is not None
        assert "comparison not possible" in result.answer_note

    @pytest.mark.asyncio
    async def test_list_op_degrades_partially_on_fx_failure(self) -> None:
        findings = AgentFindings(
            metric_requested="revenue",
            findings=(_finding("A", 100.0, "EUR"), _finding("B", 200.0, "USD")),
            comparison_op="list",
        )
        with respx.mock:
            respx.get(url__startswith=FRANKFURTER_BASE).mock(return_value=httpx.Response(500))
            result = await process_findings(findings, requested_currency="USD")

        assert result.answer_note is not None
        assert "FX conversion failed" in result.answer_note
        by_entity = {nf.finding.entity: nf for nf in result.findings}
        assert by_entity["A"].normalized_value is None  # failed conversion
        assert by_entity["B"].normalized_value == 200.0  # same currency, no conversion needed


class TestComparisonOp:
    @pytest.mark.asyncio
    async def test_argmin_picks_smallest(self) -> None:
        findings = AgentFindings(
            metric_requested="revenue",
            findings=(_finding("A", 300.0, "USD"), _finding("B", 100.0, "USD")),
            comparison_op="argmin",
        )
        result = await process_findings(findings)
        assert result.answer_entity == "B"

    @pytest.mark.asyncio
    async def test_argmax_picks_largest(self) -> None:
        findings = AgentFindings(
            metric_requested="revenue",
            findings=(_finding("A", 300.0, "USD"), _finding("B", 100.0, "USD")),
            comparison_op="argmax",
        )
        result = await process_findings(findings)
        assert result.answer_entity == "A"

    @pytest.mark.asyncio
    async def test_null_currency_excluded_from_ranking(self) -> None:
        findings = AgentFindings(
            metric_requested="revenue",
            findings=(_finding("A", 300.0, None), _finding("B", 100.0, "USD")),
            comparison_op="argmax",
        )
        result = await process_findings(findings)
        assert result.answer_entity == "B"
        assert result.answer_note is not None
        assert "excluded from ranking" in result.answer_note

    @pytest.mark.asyncio
    async def test_only_one_available_entity_note(self) -> None:
        findings = AgentFindings(
            metric_requested="revenue",
            findings=(
                _finding("A", 100.0, "USD"),
                _finding("B", None, None, available=False, reason="not found"),
            ),
            comparison_op="none",
        )
        result = await process_findings(findings)
        assert result.answer_note == "only one entity had available data"
