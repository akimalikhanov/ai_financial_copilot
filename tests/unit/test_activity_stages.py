"""Stage-row templates: `where`, `route_stage`, `agent_stage`, `scan_stage`."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from src.schemas.query_router import DocumentScopeResult, EntityManifestItem, RouterOutput
from src.services.chat.activity import (
    StageResult,
    agent_stage,
    route_stage,
    scan_stage,
    where,
)
from src.services.chat.agent.state import AgentLoopMeta

_RBA = "Ritchie Bros. Auctioneers Incorporated"


def _manifest(name: str, *years: int | None) -> EntityManifestItem:
    return EntityManifestItem(
        entity_name=name,
        doc_summaries=[{"doc_id": str(uuid4()), "name": "10-K", "year": y} for y in years],
    )


def _companies(*names: str, missing: tuple[str, ...] = (), **years: Any) -> DocumentScopeResult:
    per_entity = {n: [uuid4()] for n in names} | {n: [] for n in missing}
    return DocumentScopeResult(
        doc_ids=[d for ids in per_entity.values() for d in ids],
        source="entity_resolved",
        per_entity_doc_ids=per_entity,
        entity_manifest=[_manifest(n, *years.get("years", ())) for n in names],
    )


def _docs(source: str, n: int) -> DocumentScopeResult:
    return DocumentScopeResult(doc_ids=[uuid4() for _ in range(n)], source=source)  # type: ignore[arg-type]


def _router(route: str = "retrieval", shape: str | None = "extraction") -> RouterOutput:
    return RouterOutput(
        route=route,  # type: ignore[arg-type]
        entities=[],
        user_intent="t",
        reasoning="t",
        query_shape=shape,  # type: ignore[arg-type]
    )


def _meta(**kw: Any) -> AgentLoopMeta:
    fields: dict[str, Any] = {
        "iterations": 3,
        "tool_calls_total": 9,
        "report_calls_total": 2,
        "convergence_reason": "covered",
        "sealed": True,
    }
    fields.update(kw)
    return AgentLoopMeta(**fields)


class TestWhere:
    @pytest.mark.parametrize(
        ("scope", "text"),
        [
            (_companies(_RBA, years=(2022, 2023, None)), f"{_RBA} (2023, 2022)"),
            (_companies(_RBA), _RBA),
            (_companies("Apple Inc.", "SAP SE"), "Apple Inc. and SAP SE"),
            (_companies("A", "B", "C"), "3 companies"),
            (_companies("A", missing=("SAP SE",)), "A"),
            (_docs("explicit", 4), "4 selected documents"),
            (_docs("filtered", 1), "1 filtered document"),
            (_docs("all", 14), "all 14 documents"),
            (None, "your documents"),
        ],
        ids=[
            "one_company_years",
            "one_company_no_years",
            "two",
            "three",
            "missing_not_counted",
            "explicit",
            "filtered",
            "all",
            "no_scope",
        ],
    )
    def test_where(self, scope: DocumentScopeResult | None, text: str) -> None:
        assert where(scope) == text


class TestRouteStage:
    def test_extraction_one_company(self) -> None:
        scope = _companies(_RBA, years=(2022, 2023))
        assert route_stage(_router(), scope) == StageResult(note=f"{_RBA} · 2023, 2022")

    def test_extraction_without_companies(self) -> None:
        assert route_stage(_router(), _docs("all", 14)) == StageResult(note="all 14 documents")

    def test_comparison_lists_every_company(self) -> None:
        scope = _companies(
            "Apple Inc.", "Microsoft Corporation", missing=("SAP SE",), years=(2023,)
        )
        result = route_stage(_router(shape="comparison"), scope)
        assert result.note == "3 companies"
        assert result.companies == [
            {"name": "Apple Inc.", "years": [2023], "found": True},
            {"name": "Microsoft Corporation", "years": [2023], "found": True},
            {"name": "SAP SE", "years": [], "found": False},
        ]

    @pytest.mark.parametrize(
        ("scope", "note"),
        [
            (_companies(_RBA), f"explain · {_RBA}"),
            (_docs("all", 14), "explain · all 14 documents"),
        ],
    )
    def test_analytical(self, scope: DocumentScopeResult, note: str) -> None:
        assert route_stage(_router(shape="analytical"), scope) == StageResult(note=note)

    def test_direct_answer(self) -> None:
        result = route_stage(_router("direct_answer", None), None)
        assert result == StageResult(note="answer without searching")

    def test_out_of_scope_has_no_note(self) -> None:
        assert route_stage(_router("out_of_scope", None), None) == StageResult()


class TestAgentStage:
    def test_covered(self) -> None:
        assert agent_stage(_meta()) == StageResult("ok", "7 searches, 3 passes")

    def test_singular(self) -> None:
        meta = _meta(iterations=1, tool_calls_total=1, report_calls_total=0)
        assert agent_stage(meta).note == "1 search, 1 pass"

    @pytest.mark.parametrize(
        ("kw", "status"),
        [
            ({"convergence_reason": "search_unavailable", "sealed": False}, "fail"),
            ({"convergence_reason": "iteration_cap", "sealed": False}, "warn"),
            ({"convergence_reason": "deadline"}, "warn"),
            ({"degraded_capabilities": frozenset({"rerank"})}, "warn"),
            ({"convergence_reason": "iteration_cap"}, "ok"),
        ],
        ids=["search_unavailable", "not_sealed", "deadline", "degraded", "cap_but_sealed"],
    )
    def test_status(self, kw: dict[str, Any], status: str) -> None:
        assert agent_stage(_meta(**kw)).status == status


class TestScanStage:
    def test_clean_stays_hidden(self) -> None:
        assert scan_stage("clean") is None

    def test_flag_shows_a_warning(self) -> None:
        assert scan_stage("flag") == StageResult(
            "warn", "removed suspicious instructions", hidden=False
        )

    def test_block_fails_visibly(self) -> None:
        assert scan_stage("block") == StageResult("fail", hidden=False)


def test_detail_falls_back_to_the_stage_default_visibility() -> None:
    assert StageResult().to_detail(True) == {"status": "ok", "hidden": True}
    assert StageResult("fail", hidden=False).to_detail(True) == {"status": "fail", "hidden": False}
