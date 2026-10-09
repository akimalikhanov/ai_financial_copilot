"""Unit tests for the scope_clarification payload and its plain-text form."""

from __future__ import annotations

from uuid import uuid4

from src.schemas.query_router import CompanyCandidate, DocumentScopeResult, EntityClarification
from src.services.chat.events import build_scope_clarification_event, clarification_text


def _candidate(name: str) -> CompanyCandidate:
    return CompanyCandidate(
        company_norm=name.lower(), display_name=name, doc_ids=[uuid4()], years=[2023], titles=[]
    )


def test_entity_card_lists_unresolved_and_resolved() -> None:
    rwe_id = uuid4()
    innovation = _candidate("Aurora Innovation")
    scope = DocumentScopeResult(
        doc_ids=[rwe_id, *innovation.doc_ids],
        source="entity_resolved",
        # The ambiguous "aurora" provisionally covers its top candidate; not shown as matched.
        per_entity_doc_ids={
            "RWE AG": [rwe_id],
            "Aurora Innovation": innovation.doc_ids,
            "Tesla": [],
        },
        unresolved_entities=["Tesla"],
        clarifications=[
            EntityClarification(
                entity="Aurora",
                raw_span="aurora",
                outcome="ambiguous",
                candidates=[innovation, _candidate("Aurora Mobile")],
            ),
            EntityClarification(entity="Tesla", raw_span="Tesla", outcome="none"),
        ],
    )
    payload = build_scope_clarification_event(uuid4(), scope, named_companies=True, max_companies=5)
    assert payload["outcome"] == "entities"
    assert payload["resolved"] == ["RWE AG"]
    assert [u["raw_span"] for u in payload["unresolved"]] == ["aurora", "Tesla"]
    assert payload["unresolved"][0]["candidates"][0] == {
        "company": "Aurora Innovation",
        "years": [2023],
        "doc_count": 1,
    }
    assert clarification_text(payload) == (
        'I couldn\'t tell which company you mean by "aurora".\n\nI found no document for "Tesla".'
    )


def test_too_broad_card_text_depends_on_whether_companies_were_named() -> None:
    scope = DocumentScopeResult(doc_ids=[], source="all", too_broad_count=80)
    unnamed = build_scope_clarification_event(
        uuid4(), scope, named_companies=False, max_companies=5
    )
    named = build_scope_clarification_event(uuid4(), scope, named_companies=True, max_companies=5)
    assert unnamed["outcome"] == "too_broad" and unnamed["covered_count"] == 80
    assert "This covers 80 companies" in clarification_text(unnamed)
    assert "Split it into smaller questions" in clarification_text(named)
