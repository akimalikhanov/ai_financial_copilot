"""Unit tests for scope_resolver::resolve_scope: universe, combine table and company limit
(fake DocumentRepository, fake entity resolution)."""

from __future__ import annotations

from typing import Literal, cast
from uuid import UUID, uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.schemas.query_router import (
    ChatScope,
    CompanyCandidate,
    ExtractedEntity,
    RouterOutput,
    ScopeFilters,
)
from src.services.router import scope_resolver
from src.services.router.disambiguator import Decision
from src.services.router.entity_resolver import EntityResolution
from src.utils.config import get_scope_max_companies

# DocumentRepository is fully replaced by FakeRepo in these tests, so the real
# session is never touched — a plain object stands in, cast only to satisfy typing.
_FAKE_SESSION = cast(AsyncSession, object())

Doc = tuple[UUID, str | None, str | None, int | None]


def _doc(norm: str | None, company: str | None = None, year: int | None = 2023) -> Doc:
    return (uuid4(), norm, company or (norm.title() if norm else None), year)


AURORA = _doc("aurora innovation", "Aurora Innovation, Inc.", 2022)
RWE = _doc("rwe", "RWE AG")
NO_COMPANY = _doc(None)


def _candidate(doc: Doc) -> CompanyCandidate:
    assert doc[1] is not None and doc[2] is not None
    return CompanyCandidate(
        company_norm=doc[1], display_name=doc[2], doc_ids=[doc[0]], years=[], titles=[]
    )


class FakeRepo:
    def __init__(self, docs: list[Doc], filtered: list[UUID] | None = None) -> None:
        self.docs = docs
        self.filtered = filtered or []

    async def get_scope_docs(self, user_id, doc_ids=None):  # noqa: ARG002
        return [d for d in self.docs if doc_ids is None or d[0] in doc_ids]

    async def find_by_metadata_filters(self, user_id, **_kwargs):  # noqa: ARG002
        return self.filtered


def _entity(name: str) -> ExtractedEntity:
    return ExtractedEntity(name=name, entity_type="company", raw_span=name)


def _router_output(*entities: ExtractedEntity) -> RouterOutput:
    return RouterOutput(route="retrieval", entities=list(entities), user_intent="x", reasoning="y")


@pytest.fixture
def setup(monkeypatch: pytest.MonkeyPatch):
    def _make(docs: list[Doc], resolutions=(), filtered=None) -> None:
        repo = FakeRepo(docs, filtered)
        monkeypatch.setattr(scope_resolver, "DocumentRepository", lambda _session: repo)

        async def fake_resolve(*_args, **_kwargs):
            return list(resolutions)

        monkeypatch.setattr(scope_resolver, "resolve_entities", fake_resolve)

    return _make


async def _scope(scope: ChatScope | None, *entities: ExtractedEntity):
    return await scope_resolver.resolve_scope(
        _FAKE_SESSION, uuid4(), scope, _router_output(*entities)
    )


def _resolved(doc: Doc, decision: Decision = "resolved") -> EntityResolution:
    return EntityResolution(decision=decision, candidates=[_candidate(doc)], method="fast_path")


_NONE = EntityResolution(decision="none", candidates=[], method="llm")


class TestUniverse:
    @pytest.mark.asyncio
    async def test_no_entities_covers_every_company_with_explicit_ids(self, setup) -> None:
        setup([AURORA, RWE])
        result = await _scope(None)
        assert result.source == "all"
        assert result.doc_ids == [AURORA[0], RWE[0]]
        assert result.per_entity_doc_ids == {
            "Aurora Innovation, Inc.": [AURORA[0]],
            "RWE AG": [RWE[0]],
        }
        assert result.entity_manifest is not None
        assert result.entity_manifest[0].doc_summaries[0]["year"] == 2022

    @pytest.mark.asyncio
    async def test_documents_without_a_company_form_one_group(self, setup) -> None:
        other = _doc(None)
        setup([NO_COMPANY, other])
        result = await _scope(None)
        assert result.per_entity_doc_ids == {scope_resolver.NO_COMPANY: [NO_COMPANY[0], other[0]]}

    @pytest.mark.asyncio
    async def test_selection_is_the_universe(self, setup) -> None:
        setup([AURORA, RWE])
        result = await _scope(ChatScope(mode="selectedDocs", doc_ids=[RWE[0]]))
        assert result.source == "explicit"
        assert result.doc_ids == [RWE[0]]

    @pytest.mark.asyncio
    async def test_empty_selection_counts_as_all_documents(self, setup) -> None:
        setup([AURORA, RWE])
        result = await _scope(ChatScope(mode="selectedDocs", doc_ids=[]))
        assert result.source == "all"
        assert result.doc_ids == [AURORA[0], RWE[0]]

    @pytest.mark.asyncio
    async def test_filter_matching_nothing_searches_nothing(self, setup) -> None:
        setup([AURORA], filtered=[])
        scope = ChatScope(mode="filteredByMetadata", filters=ScopeFilters(year=[1999]))
        result = await _scope(scope)
        assert result.source == "filtered"
        assert result.doc_ids == []
        assert result.per_entity_doc_ids is None


class TestCombine:
    @pytest.mark.asyncio
    async def test_resolved_inside_universe_is_keyed_by_display_name(self, setup) -> None:
        setup([AURORA, RWE], [_resolved(AURORA)])
        result = await _scope(None, _entity("Aurora"))
        assert result.source == "entity_resolved"
        assert result.doc_ids == [AURORA[0]]
        assert result.per_entity_doc_ids == {"Aurora Innovation, Inc.": [AURORA[0]]}
        assert result.clarifications == []

    @pytest.mark.asyncio
    async def test_resolved_outside_universe_is_outside_scope(self, setup) -> None:
        setup([AURORA, RWE], [_resolved(RWE)])
        result = await _scope(ChatScope(mode="selectedDocs", doc_ids=[AURORA[0]]), _entity("RWE"))
        assert result.source == "unresolved"
        assert result.doc_ids == []
        assert result.per_entity_doc_ids == {"RWE": []}
        assert result.unresolved_entities == ["RWE"]
        assert [c.outcome for c in result.clarifications] == ["outside_scope"]

    @pytest.mark.asyncio
    async def test_none_is_not_found(self, setup) -> None:
        setup([AURORA], [_NONE])
        result = await _scope(None, _entity("Tesla"))
        assert result.per_entity_doc_ids == {"Tesla": []}
        assert result.unresolved_entities == ["Tesla"]
        assert [c.outcome for c in result.clarifications] == ["none"]

    @pytest.mark.asyncio
    async def test_ambiguous_uses_first_candidate_and_is_flagged(self, setup) -> None:
        setup([AURORA], [_resolved(AURORA, decision="ambiguous")])
        result = await _scope(None, _entity("Aurora"))
        assert result.per_entity_doc_ids == {"Aurora Innovation, Inc.": [AURORA[0]]}
        assert result.unresolved_entities == []
        assert [c.outcome for c in result.clarifications] == ["ambiguous"]

    @pytest.mark.asyncio
    async def test_partial_match_keeps_the_resolved_entity(self, setup) -> None:
        setup([AURORA, RWE], [_NONE, _resolved(RWE)])
        result = await _scope(None, _entity("Tesla"), _entity("RWE"))
        assert result.source == "entity_resolved"
        assert result.doc_ids == [RWE[0]]
        assert result.per_entity_doc_ids == {"RWE AG": [RWE[0]], "Tesla": []}


class TestCompanyLimit:
    @pytest.mark.parametrize(
        "mode", ["allDocs", "selectedDocs", "filteredByMetadata"], ids=lambda m: m
    )
    @pytest.mark.asyncio
    async def test_too_many_companies_is_too_broad_in_every_mode(
        self, setup, monkeypatch: pytest.MonkeyPatch, mode: str
    ) -> None:
        monkeypatch.setenv("SCOPE_MAX_COMPANIES", "2")
        docs = [_doc(f"company {i}") for i in range(3)]
        ids = [d[0] for d in docs]
        setup(docs, filtered=ids)
        scope = ChatScope(mode=mode, doc_ids=ids)  # type: ignore[arg-type]
        result = await _scope(scope)
        assert result.too_broad_count == 3
        assert result.doc_ids == []
        assert result.per_entity_doc_ids is None

    @pytest.mark.asyncio
    async def test_at_the_limit_is_not_too_broad(
        self, setup, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SCOPE_MAX_COMPANIES", "2")
        setup([AURORA, RWE])
        result = await _scope(None)
        assert result.too_broad_count is None

    @pytest.mark.asyncio
    async def test_named_companies_count_too(self, setup, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SCOPE_MAX_COMPANIES", "1")
        setup([AURORA, RWE], [_resolved(AURORA), _resolved(RWE)])
        result = await _scope(None, _entity("Aurora"), _entity("RWE"))
        assert result.too_broad_count == 2

    def test_limit_above_plan_items_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENT_MAX_PLAN_ITEMS", "4")
        monkeypatch.setenv("SCOPE_MAX_COMPANIES", "5")
        with pytest.raises(ValueError, match="AGENT_MAX_PLAN_ITEMS"):
            get_scope_max_companies()


class TestClarificationTriggers:
    @pytest.mark.asyncio
    async def test_only_the_unresolved_entities_are_asked_about(self, setup) -> None:
        setup([AURORA, RWE], [_NONE, _resolved(RWE)])
        result = await _scope(None, _entity("Tesla"), _entity("RWE"))
        assert [c.raw_span for c in result.clarifications] == ["Tesla"]

    @pytest.mark.asyncio
    async def test_a_person_is_not_asked_about(self, setup) -> None:
        setup([AURORA], [_NONE])
        person = ExtractedEntity(name="Satya Nadella", entity_type="person", raw_span="Nadella")
        result = await _scope(None, person)
        assert result.unresolved_entities == ["Satya Nadella"]
        assert result.clarifications == []

    @pytest.mark.asyncio
    async def test_no_card_for_a_question_without_a_company_inside_the_limit(self, setup) -> None:
        setup([AURORA, RWE])
        result = await _scope(None)
        assert result.clarifications == [] and result.too_broad_count is None

    @pytest.mark.asyncio
    async def test_binding_include_searches_outside_the_scope(self, setup) -> None:
        setup([AURORA, RWE], [_bound(RWE, "include")])
        result = await _scope(ChatScope(mode="selectedDocs", doc_ids=[AURORA[0]]), _entity("RWE"))
        assert result.per_entity_doc_ids == {"RWE AG": [RWE[0]]}
        assert result.clarifications == []

    @pytest.mark.asyncio
    async def test_binding_exclude_leaves_it_out_without_asking(self, setup) -> None:
        setup([AURORA, RWE], [_bound(RWE, "exclude")])
        result = await _scope(ChatScope(mode="selectedDocs", doc_ids=[AURORA[0]]), _entity("RWE"))
        assert result.unresolved_entities == ["RWE"]
        assert result.clarifications == []


def _bound(doc: Doc, outside_scope: Literal["include", "exclude"]) -> EntityResolution:
    return EntityResolution(
        decision="resolved",
        candidates=[_candidate(doc)],
        method="binding",
        outside_scope=outside_scope,
    )
