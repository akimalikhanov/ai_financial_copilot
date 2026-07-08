"""Unit tests for scope_resolver::resolve_scope (mocked session/DocumentRepository)."""

from __future__ import annotations

from typing import cast
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.schemas.query_router import ChatScope, ExtractedEntity, RouterOutput, ScopeFilters
from src.services.router import scope_resolver

# DocumentRepository is fully replaced by FakeRepo in these tests, so the real
# session is never touched — a plain object stands in, cast only to satisfy typing.
_FAKE_SESSION = cast(AsyncSession, object())


class FakeRepo:
    def __init__(
        self,
        metadata_filter_result: list | None = None,
        doc_summaries: list[tuple] | None = None,
    ) -> None:
        self._metadata_filter_result = metadata_filter_result or []
        self._doc_summaries = doc_summaries or []

    async def find_by_metadata_filters(self, user_id, **_kwargs):  # noqa: ARG002
        return self._metadata_filter_result

    async def get_scope_doc_summaries(self, user_id, doc_ids, **_kwargs):  # noqa: ARG002
        return [row for row in self._doc_summaries if row[0] in doc_ids]


def _router_output(entities: list[ExtractedEntity] | None = None) -> RouterOutput:
    return RouterOutput(route="retrieval", entities=entities or [], user_intent="x", reasoning="y")


def _entity(name: str) -> ExtractedEntity:
    return ExtractedEntity(name=name, entity_type="company", raw_span=name)


@pytest.fixture(autouse=True)
def _patch_repo(monkeypatch: pytest.MonkeyPatch):
    def _make(**kwargs):
        repo = FakeRepo(**kwargs)
        monkeypatch.setattr(scope_resolver, "DocumentRepository", lambda _session: repo)
        return repo

    return _make


class TestSelectedDocsExplicit:
    @pytest.mark.asyncio
    async def test_explicit_scope_with_doc_ids_short_circuits(self, _patch_repo) -> None:
        doc_id = uuid4()
        _patch_repo(doc_summaries=[(doc_id, "Acme", 2023)])
        scope = ChatScope(mode="selectedDocs", doc_ids=[doc_id])
        result = await scope_resolver.resolve_scope(_FAKE_SESSION, uuid4(), scope, _router_output())
        assert result.source == "explicit"
        assert result.doc_ids == [doc_id]

    @pytest.mark.asyncio
    async def test_empty_doc_ids_falls_through_to_alldocs(self, _patch_repo) -> None:
        _patch_repo()
        scope = ChatScope(mode="selectedDocs", doc_ids=[])
        result = await scope_resolver.resolve_scope(_FAKE_SESSION, uuid4(), scope, _router_output())
        assert result.source == "all"
        assert result.doc_ids is None


class TestFilteredByMetadata:
    @pytest.mark.asyncio
    async def test_small_result_set_skips_layer2(self, _patch_repo) -> None:
        doc_ids = [uuid4(), uuid4()]  # <= filtered_md_thresh (5)
        _patch_repo(
            metadata_filter_result=doc_ids,
            doc_summaries=[(d, "Acme", 2023) for d in doc_ids],
        )
        scope = ChatScope(mode="filteredByMetadata", filters=ScopeFilters(company=["Acme"]))
        result = await scope_resolver.resolve_scope(_FAKE_SESSION, uuid4(), scope, _router_output())
        assert result.source == "filtered"
        assert result.doc_ids is not None
        assert set(result.doc_ids) == set(doc_ids)

    @pytest.mark.asyncio
    async def test_small_result_set_empty_becomes_none(self, _patch_repo) -> None:
        _patch_repo(metadata_filter_result=[])
        scope = ChatScope(mode="filteredByMetadata", filters=ScopeFilters(company=["Nope"]))
        result = await scope_resolver.resolve_scope(_FAKE_SESSION, uuid4(), scope, _router_output())
        assert result.source == "filtered"
        assert result.doc_ids is None

    @pytest.mark.asyncio
    async def test_large_result_set_with_entities_intersects(
        self, monkeypatch: pytest.MonkeyPatch, _patch_repo
    ) -> None:
        layer1_ids = [uuid4() for _ in range(6)]  # > filtered_md_thresh (5)
        _patch_repo(metadata_filter_result=layer1_ids)

        matched_id = layer1_ids[0]

        async def fake_resolve_all(*_args, constrain_to, **_kwargs):
            assert constrain_to == layer1_ids
            return {"Acme": [matched_id]}

        monkeypatch.setattr(scope_resolver, "_resolve_all_entities", fake_resolve_all)

        scope = ChatScope(mode="filteredByMetadata", filters=ScopeFilters(company=["Acme"]))
        result = await scope_resolver.resolve_scope(
            _FAKE_SESSION, uuid4(), scope, _router_output([_entity("Acme")])
        )
        assert result.source == "filtered"
        assert result.doc_ids == [matched_id]
        assert result.per_entity_doc_ids == {"Acme": [matched_id]}

    @pytest.mark.asyncio
    async def test_large_result_set_entity_match_empty_falls_back_to_layer1(
        self, monkeypatch: pytest.MonkeyPatch, _patch_repo
    ) -> None:
        layer1_ids = [uuid4() for _ in range(6)]
        _patch_repo(metadata_filter_result=layer1_ids)

        async def fake_resolve_all(*_args, **_kwargs):
            return {"Acme": []}

        monkeypatch.setattr(scope_resolver, "_resolve_all_entities", fake_resolve_all)

        scope = ChatScope(mode="filteredByMetadata", filters=ScopeFilters(company=["Acme"]))
        result = await scope_resolver.resolve_scope(
            _FAKE_SESSION, uuid4(), scope, _router_output([_entity("Acme")])
        )
        assert result.doc_ids == layer1_ids
        # per_entity is a non-empty dict ({"Acme": []}), so it's truthy and kept
        # as-is — only an empty dict (no entities at all) collapses to None.
        assert result.per_entity_doc_ids == {"Acme": []}

    @pytest.mark.asyncio
    async def test_large_result_set_no_entities_returns_layer1_grouped(self, _patch_repo) -> None:
        layer1_ids = [uuid4() for _ in range(6)]
        _patch_repo(
            metadata_filter_result=layer1_ids,
            doc_summaries=[(d, "Acme", 2023) for d in layer1_ids],
        )
        scope = ChatScope(mode="filteredByMetadata", filters=ScopeFilters(company=["Acme"]))
        result = await scope_resolver.resolve_scope(_FAKE_SESSION, uuid4(), scope, _router_output())
        assert result.source == "filtered"
        assert result.doc_ids == layer1_ids


class TestAllDocs:
    @pytest.mark.asyncio
    async def test_no_entities_no_scope_no_prefilter(self, _patch_repo) -> None:
        _patch_repo()
        result = await scope_resolver.resolve_scope(_FAKE_SESSION, uuid4(), None, _router_output())
        assert result.source == "all"
        assert result.doc_ids is None

    @pytest.mark.asyncio
    async def test_entities_matched_returns_entity_resolved(
        self, monkeypatch: pytest.MonkeyPatch, _patch_repo
    ) -> None:
        _patch_repo()
        matched_id = uuid4()

        async def fake_resolve_all(*_args, **_kwargs):
            return {"Acme": [matched_id]}

        monkeypatch.setattr(scope_resolver, "_resolve_all_entities", fake_resolve_all)

        result = await scope_resolver.resolve_scope(
            _FAKE_SESSION, uuid4(), None, _router_output([_entity("Acme")])
        )
        assert result.source == "entity_resolved"
        assert result.doc_ids == [matched_id]

    @pytest.mark.asyncio
    async def test_entities_unmatched_falls_back_to_all_no_per_entity(
        self, monkeypatch: pytest.MonkeyPatch, _patch_repo
    ) -> None:
        _patch_repo()

        async def fake_resolve_all(*_args, **_kwargs):
            return {"Acme": []}

        monkeypatch.setattr(scope_resolver, "_resolve_all_entities", fake_resolve_all)

        result = await scope_resolver.resolve_scope(
            _FAKE_SESSION, uuid4(), None, _router_output([_entity("Acme")])
        )
        assert result.source == "all"
        assert result.doc_ids is None
        assert result.per_entity_doc_ids is None
