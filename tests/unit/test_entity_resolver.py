"""Unit tests for entity_resolver (mocked DocumentRepository)."""

from __future__ import annotations

from typing import cast
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.schemas.query_router import ExtractedEntity
from src.services.router import entity_resolver

# DocumentRepository is fully replaced by FakeRepo in these tests, so the real
# session is never touched — a plain object stands in, cast only to satisfy typing.
_FAKE_SESSION = cast(AsyncSession, object())


class FakeRepo:
    def __init__(self, by_name: dict[str, list]) -> None:
        self._by_name = by_name
        self.calls: list[dict] = []

    async def find_by_company_similarity(self, user_id, name, *, threshold, constrain_to, limit):
        self.calls.append(
            {
                "user_id": user_id,
                "name": name,
                "threshold": threshold,
                "constrain_to": constrain_to,
                "limit": limit,
            }
        )
        return self._by_name.get(name, [])


def _entity(name: str) -> ExtractedEntity:
    return ExtractedEntity(name=name, entity_type="company", raw_span=name)


class TestResolveEntitiesToDocIds:
    @pytest.mark.asyncio
    async def test_empty_entities_short_circuits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(entity_resolver, "DocumentRepository", lambda _session: FakeRepo({}))
        matched, unresolved = await entity_resolver.resolve_entities_to_doc_ids(
            session=_FAKE_SESSION, user_id=uuid4(), entities=[]
        )
        assert matched == []
        assert unresolved == []

    @pytest.mark.asyncio
    async def test_separates_matched_and_unresolved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        doc_id = uuid4()
        repo = FakeRepo({"Acme": [doc_id], "Nope": []})
        monkeypatch.setattr(entity_resolver, "DocumentRepository", lambda _session: repo)

        matched, unresolved = await entity_resolver.resolve_entities_to_doc_ids(
            session=_FAKE_SESSION, user_id=uuid4(), entities=[_entity("Acme"), _entity("Nope")]
        )
        assert matched == [doc_id]
        assert len(unresolved) == 1
        assert unresolved[0].name == "Nope"


class TestResolveEntitiesPerEntity:
    @pytest.mark.asyncio
    async def test_empty_entities_short_circuits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(entity_resolver, "DocumentRepository", lambda _session: FakeRepo({}))
        result = await entity_resolver.resolve_entities_per_entity(
            session=_FAKE_SESSION, user_id=uuid4(), entities=[]
        )
        assert result == {}

    @pytest.mark.asyncio
    async def test_includes_empty_lists_for_unresolved(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        doc_id = uuid4()
        repo = FakeRepo({"Acme": [doc_id], "Nope": []})
        monkeypatch.setattr(entity_resolver, "DocumentRepository", lambda _session: repo)

        result = await entity_resolver.resolve_entities_per_entity(
            session=_FAKE_SESSION, user_id=uuid4(), entities=[_entity("Acme"), _entity("Nope")]
        )
        assert result == {"Acme": [doc_id], "Nope": []}
