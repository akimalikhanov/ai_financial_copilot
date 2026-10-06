"""Unit tests for entity_resolver (fake DocumentRepository and disambiguator)."""

from __future__ import annotations

from typing import cast
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.schemas.query_router import CompanyCandidate, ExtractedEntity
from src.services.router import entity_resolver
from src.services.router.disambiguator import Disambiguation

# DocumentRepository is fully replaced by FakeRepo in these tests, so the real
# session is never touched — a plain object stands in, cast only to satisfy typing.
_FAKE_SESSION = cast(AsyncSession, object())


def _company(norm: str, score: float = 0.0) -> CompanyCandidate:
    return CompanyCandidate(
        company_norm=norm,
        display_name=norm.title(),
        doc_ids=[uuid4()],
        years=[],
        titles=[],
        score=score,
    )


INNOVATION = _company("aurora innovation", 0.6)
MOBILE = _company("aurora mobile", 0.5)
RWE = _company("rwe", 1.0)


class FakeRepo:
    def __init__(self, by_key: dict[str, list[CompanyCandidate]], catalogue: list) -> None:
        self._by_key = by_key
        self._catalogue = catalogue

    async def find_company_candidates(self, user_id, keys, **_kwargs):  # noqa: ARG002
        return {k: self._by_key.get(k, []) for k in keys}

    async def list_companies(self, user_id, *, limit):  # noqa: ARG002
        return self._catalogue[:limit]


class FakeDisambiguator:
    def __init__(self, result: list[Disambiguation] | None) -> None:
        self._result = result
        self.calls: list[tuple] = []

    async def __call__(self, query, entities, candidates, **_kwargs):
        self.calls.append((query, entities, candidates))
        return self._result


def _entity(name: str, span: str | None = None) -> ExtractedEntity:
    return ExtractedEntity(name=name, entity_type="company", raw_span=span or name)


@pytest.fixture
def setup(monkeypatch: pytest.MonkeyPatch):
    def _make(by_key, catalogue=(), result=None) -> FakeDisambiguator:
        repo = FakeRepo(by_key, list(catalogue))
        fake = FakeDisambiguator(result)
        monkeypatch.setattr(entity_resolver, "DocumentRepository", lambda _session: repo)
        monkeypatch.setattr(entity_resolver, "disambiguate", fake)
        return fake

    return _make


async def _resolve(entities: list[ExtractedEntity]):
    return await entity_resolver.resolve_entities(_FAKE_SESSION, uuid4(), "q", entities)


@pytest.mark.asyncio
async def test_no_entities_returns_empty(setup) -> None:
    setup({})
    assert await _resolve([]) == []


@pytest.mark.asyncio
async def test_single_exact_match_makes_no_llm_call(setup) -> None:
    fake = setup({"rwe": [RWE]})
    [res] = await _resolve([_entity("RWE AG", "RWE")])
    assert (res.decision, res.method, res.candidates) == ("resolved", "fast_path", [RWE])
    assert fake.calls == []


@pytest.mark.asyncio
async def test_llm_decision_is_kept_with_its_ranking(setup) -> None:
    fake = setup(
        {"aurora": [INNOVATION, MOBILE]},
        catalogue=[INNOVATION, MOBILE],
        result=[Disambiguation(decision="ambiguous", candidates=[MOBILE, INNOVATION])],
    )
    [res] = await _resolve([_entity("Aurora")])
    assert (res.decision, res.method, res.candidates) == ("ambiguous", "llm", [MOBILE, INNOVATION])
    assert len(fake.calls) == 1


@pytest.mark.asyncio
async def test_failed_call_falls_back_to_trigram_candidates(setup) -> None:
    setup({"aurora": [MOBILE, INNOVATION]}, catalogue=[INNOVATION, MOBILE], result=None)
    [res] = await _resolve([_entity("Aurora")])
    assert (res.decision, res.method) == ("ambiguous", "fallback")
    assert res.candidates[0] == INNOVATION  # best trigram score first


@pytest.mark.asyncio
async def test_small_catalogue_is_the_candidate_pool(setup) -> None:
    fake = setup(
        {}, catalogue=[INNOVATION, RWE], result=[Disambiguation(decision="none", candidates=[])]
    )
    [res] = await _resolve([_entity("Microsoft Corporation", "MSFT")])
    assert fake.calls[0][2] == [INNOVATION, RWE]
    assert res.decision == "none"


@pytest.mark.asyncio
async def test_large_catalogue_sends_only_trigram_candidates(
    setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ENTITY_CATALOGUE_INLINE_MAX", "1")
    fake = setup(
        {"aurora": [INNOVATION, MOBILE]},
        catalogue=[INNOVATION, MOBILE, RWE],
        result=[Disambiguation(decision="resolved", candidates=[INNOVATION])],
    )
    await _resolve([_entity("Aurora")])
    assert fake.calls[0][2] == [INNOVATION, MOBILE]


@pytest.mark.asyncio
async def test_binding_resolves_without_llm(setup, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = setup({"aurora innovation": [INNOVATION]}, catalogue=[INNOVATION, MOBILE])

    class FakeConversations:
        def __init__(self, _session) -> None:
            pass

        async def get_entity_bindings(self, _conversation_id):
            return {"aurora": {"company_norm": "aurora innovation"}}

    monkeypatch.setattr(entity_resolver, "ConversationRepository", FakeConversations)
    [res] = await entity_resolver.resolve_entities(
        _FAKE_SESSION, uuid4(), "q", [_entity("Aurora")], conversation_id=uuid4()
    )
    assert (res.decision, res.method, res.candidates) == ("resolved", "binding", [INNOVATION])
    assert fake.calls == []


@pytest.mark.asyncio
async def test_no_candidates_at_all_skips_the_llm(setup) -> None:
    fake = setup({})
    [res] = await _resolve([_entity("Tesla")])
    assert (res.decision, res.method) == ("none", "no_candidates")
    assert fake.calls == []
