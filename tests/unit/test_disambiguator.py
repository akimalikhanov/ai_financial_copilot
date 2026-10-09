"""Unit tests for the entity disambiguator (fake LLM, fake sub-request repository)."""

from __future__ import annotations

import asyncio
import json
from typing import cast
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.schemas.query_router import CompanyCandidate, ExtractedEntity
from src.services.llm_adapters.base_adapter import LLMResponse, LLMResponseStats
from src.services.llm_router import LLMRouter
from src.services.router import disambiguator
from src.services.router.disambiguator import _response_format, disambiguate


def _fake_session() -> AsyncMock:
    """An AsyncSession double whose begin_nested() works as `async with`."""
    session = AsyncMock()
    session.begin_nested = MagicMock(return_value=AsyncMock())
    return session


_FAKE_SESSION = cast(AsyncSession, _fake_session())


class FakeLLM:
    provider = "fake"

    def __init__(self, text: str = "", delay: float = 0.0, events: list[str] | None = None) -> None:
        self._text = text
        self._delay = delay
        self._events = events
        self.calls: list[dict] = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        if self._events is not None:
            self._events.append("llm")
        await asyncio.sleep(self._delay)
        return LLMResponse(
            text=self._text, stats=LLMResponseStats(input_tokens=120, output_tokens=20)
        )


class _FakeRouter:
    def __init__(self, llm: FakeLLM) -> None:
        self._llm = llm

    def get(self, _model_id: str) -> FakeLLM:
        return self._llm


class FakeRequestRepo:
    rows: list[dict] = []

    def __init__(self, _session) -> None:
        pass

    async def create_subrequest(self, **kwargs) -> None:
        FakeRequestRepo.rows.append(kwargs)


@pytest.fixture(autouse=True)
def _repo(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    FakeRequestRepo.rows = []
    monkeypatch.setattr(disambiguator, "LLMRequestRepository", FakeRequestRepo)
    return FakeRequestRepo.rows


def _company(name: str) -> CompanyCandidate:
    return CompanyCandidate(
        company_norm=name.lower(), display_name=name, doc_ids=[uuid4()], years=[2022], titles=[]
    )


AURORAS = [_company("Aurora Innovation"), _company("Aurora Mobile")]
ENTITIES = [ExtractedEntity(name="Aurora", entity_type="company", raw_span="aurora")]


async def _run(llm: FakeLLM, **kwargs):
    return await disambiguate(
        "aurora revenue",
        ENTITIES,
        AURORAS,
        llm_router=cast(LLMRouter, _FakeRouter(llm)),
        **kwargs,
    )


def test_schema_enumerates_entity_and_candidate_ids() -> None:
    item = _response_format(2, 3)["json_schema"]["schema"]["properties"]["entities"]["items"]
    assert item["properties"]["entity"]["enum"] == ["E1", "E2"]
    assert item["properties"]["candidates"]["items"]["enum"] == ["C1", "C2", "C3"]
    assert item["additionalProperties"] is False


@pytest.mark.asyncio
async def test_ids_map_back_to_companies_in_order() -> None:
    llm = FakeLLM(
        json.dumps(
            {"entities": [{"entity": "E1", "decision": "ambiguous", "candidates": ["C2", "C1"]}]}
        )
    )
    [result] = await _run(llm) or []
    assert result.decision == "ambiguous"
    assert [c.display_name for c in result.candidates] == ["Aurora Mobile", "Aurora Innovation"]


@pytest.mark.asyncio
async def test_missing_entity_or_unknown_id_is_none() -> None:
    llm = FakeLLM(
        json.dumps({"entities": [{"entity": "E1", "decision": "resolved", "candidates": ["C9"]}]})
    )
    [result] = await _run(llm) or []
    assert result.decision == "none" and result.candidates == []


@pytest.mark.asyncio
async def test_success_logs_one_completed_subrequest(_repo: list[dict]) -> None:
    llm = FakeLLM(
        json.dumps({"entities": [{"entity": "E1", "decision": "resolved", "candidates": ["C1"]}]})
    )
    await _run(llm, session=_FAKE_SESSION, parent_request_id=uuid4(), conversation_id=uuid4())
    [row] = _repo
    assert row["request_type"] == "entity_disambiguator"
    assert row["status"] == "completed"
    assert row["prompt_tokens"] == 120 and row["completion_tokens"] == 20
    assert llm.calls[0]["_lf_name"] == "entity_disambiguator"


@pytest.mark.asyncio
async def test_commits_before_the_llm_call() -> None:
    """The candidate reads leave a transaction open; it must not be held across the call."""
    events: list[str] = []
    session = _fake_session()
    session.commit.side_effect = lambda: events.append("commit")
    llm = FakeLLM(
        json.dumps({"entities": [{"entity": "E1", "decision": "resolved", "candidates": ["C1"]}]}),
        events=events,
    )
    await _run(llm, session=cast(AsyncSession, session))
    assert events == ["commit", "llm"]


@pytest.mark.asyncio
async def test_failed_log_write_does_not_fail_the_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed sub-request INSERT rolls back its savepoint only; the caller's session is
    never rolled back and the decision still comes through."""

    class BrokenRepo:
        def __init__(self, _session) -> None:
            pass

        async def create_subrequest(self, **_kwargs) -> None:
            raise RuntimeError("insert failed")

    monkeypatch.setattr(disambiguator, "LLMRequestRepository", BrokenRepo)
    session = _fake_session()
    llm = FakeLLM(
        json.dumps({"entities": [{"entity": "E1", "decision": "resolved", "candidates": ["C1"]}]})
    )
    [result] = (
        await _run(
            llm,
            session=cast(AsyncSession, session),
            parent_request_id=uuid4(),
            conversation_id=uuid4(),
        )
        or []
    )
    assert result.decision == "resolved"
    session.rollback.assert_not_awaited()


@pytest.mark.asyncio
async def test_timeout_logs_one_failed_subrequest_and_returns_none(
    _repo: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ENTITY_DISAMBIGUATOR_TIMEOUT", "0.01")
    result = await _run(
        FakeLLM(delay=1.0),
        session=_FAKE_SESSION,
        parent_request_id=uuid4(),
        conversation_id=uuid4(),
    )
    assert result is None
    [row] = _repo
    assert row["status"] == "failed" and row["error_code"] == "TimeoutError"


@pytest.mark.asyncio
async def test_unparseable_output_returns_none() -> None:
    assert await _run(FakeLLM("not json")) is None
