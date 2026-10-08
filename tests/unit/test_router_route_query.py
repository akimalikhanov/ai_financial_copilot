"""Unit tests for route_query (DI via llm_router param)."""

from __future__ import annotations

import json
from typing import cast
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.schemas.query_router import ChatScope, RouterInput
from src.services.llm_adapters.base_adapter import LLMResponse
from src.services.llm_router import LLMRouter
from src.services.router import scope_resolver
from src.services.router.router import _FALLBACK, route_query


class FakeLLM:
    def __init__(self, responses: list[LLMResponse] | None = None, raises: Exception | None = None):
        self._responses = responses or []
        self._raises = raises
        self.provider = "fake"
        self.calls: list[dict] = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        if self._raises is not None:
            raise self._raises
        return self._responses[len(self.calls) - 1]


class _FakeRouterImpl:
    def __init__(self, llm: FakeLLM | Exception):
        self._llm = llm

    def get(self, _model_id: str):
        if isinstance(self._llm, Exception):
            raise self._llm
        return self._llm


def FakeRouter(llm: FakeLLM | Exception) -> LLMRouter:  # noqa: N802
    """Build a duck-typed router double, cast to LLMRouter to satisfy route_query's signature."""
    return cast(LLMRouter, _FakeRouterImpl(llm))


def _fake_session() -> AsyncMock:
    """An AsyncSession double whose begin_nested() works as `async with`."""
    session = AsyncMock()
    session.begin_nested = MagicMock(return_value=AsyncMock())
    return session


def _valid_json(route: str = "retrieval") -> str:
    return json.dumps(
        {
            "route": route,
            "entities": [],
            "user_intent": "asking about revenue",
            "reasoning": "wants a figure",
        }
    )


class TestFallbackSites:
    @pytest.mark.asyncio
    async def test_model_unavailable_returns_fallback(self) -> None:
        router = FakeRouter(RuntimeError("no such model"))
        output, scope = await route_query(RouterInput(query="hello"), llm_router=router)
        assert output == _FALLBACK
        assert scope is None

    @pytest.mark.asyncio
    async def test_llm_raises_returns_fallback(self) -> None:
        llm = FakeLLM(raises=RuntimeError("boom"))
        router = FakeRouter(llm)
        output, scope = await route_query(RouterInput(query="hello"), llm_router=router)
        assert output == _FALLBACK
        assert scope is None

    @pytest.mark.asyncio
    async def test_unparseable_response_both_attempts_returns_fallback(self) -> None:
        llm = FakeLLM(responses=[LLMResponse(text="not json"), LLMResponse(text="still not json")])
        router = FakeRouter(llm)
        output, scope = await route_query(RouterInput(query="hello"), llm_router=router)
        assert output == _FALLBACK
        assert scope is None
        assert len(llm.calls) == 2

    @pytest.mark.asyncio
    async def test_schema_validation_fails_after_retry_returns_fallback(self) -> None:
        bad = json.dumps({"foo": "bar"})
        llm = FakeLLM(responses=[LLMResponse(text=bad), LLMResponse(text=bad)])
        router = FakeRouter(llm)
        output, scope = await route_query(RouterInput(query="hello"), llm_router=router)
        assert output == _FALLBACK
        assert len(llm.calls) == 2

    @pytest.mark.asyncio
    async def test_timeout_keeps_selected_docs_scope(self, monkeypatch: pytest.MonkeyPatch) -> None:
        doc_id = uuid4()

        class FakeRepo:
            async def get_scope_docs(self, _user_id, doc_ids=None):
                return [(d, "acme", "Acme", 2023) for d in doc_ids or []]

        monkeypatch.setattr(scope_resolver, "DocumentRepository", lambda _s: FakeRepo())
        router = FakeRouter(FakeLLM(raises=TimeoutError()))
        inp = RouterInput(query="revenue?", scope=ChatScope(mode="selectedDocs", doc_ids=[doc_id]))

        output, scope = await route_query(
            inp, llm_router=router, session=cast(AsyncSession, object()), user_id=uuid4()
        )

        assert output == _FALLBACK
        assert scope is not None
        assert scope.doc_ids == [doc_id]

    @pytest.mark.asyncio
    async def test_empty_query_raises_value_error(self) -> None:
        router = FakeRouter(FakeLLM())
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await route_query(RouterInput(query="   "), llm_router=router)


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_valid_response_first_attempt(self) -> None:
        llm = FakeLLM(responses=[LLMResponse(text=_valid_json())])
        router = FakeRouter(llm)
        output, scope = await route_query(RouterInput(query="What was revenue?"), llm_router=router)
        assert output.route == "retrieval"
        assert output.user_intent == "asking about revenue"
        assert scope is None  # no session provided
        assert len(llm.calls) == 1

    @pytest.mark.asyncio
    async def test_logged_row_is_committed_before_the_retry_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The sub-request flush opens a transaction; it must not be held across the
        parse retry's LLM call."""
        from src.services.router import router as router_module

        events: list[str] = []

        class FakeRepo:
            def __init__(self, _session) -> None:
                pass

            async def create_subrequest(self, **_kwargs) -> None:
                events.append("row")

        monkeypatch.setattr(router_module, "LLMRequestRepository", FakeRepo)
        session = _fake_session()
        session.commit.side_effect = lambda: events.append("commit")

        class RecordingLLM(FakeLLM):
            async def complete(self, **kwargs):
                events.append("llm")
                return await super().complete(**kwargs)

        llm = RecordingLLM(
            responses=[LLMResponse(text="garbage"), LLMResponse(text=_valid_json("direct_answer"))]
        )
        await route_query(
            RouterInput(query="What was revenue?"),
            llm_router=FakeRouter(llm),
            session=cast(AsyncSession, session),
            parent_request_id=uuid4(),
            conversation_id=uuid4(),
        )
        assert events == ["llm", "row", "commit", "llm", "row", "commit"]

    @pytest.mark.asyncio
    async def test_failed_log_write_still_routes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A failed sub-request INSERT rolls back its savepoint only: routing goes on, the
        caller's session is never rolled back, and the connection is still released."""
        from src.services.router import router as router_module

        class BrokenRepo:
            def __init__(self, _session) -> None:
                pass

            async def create_subrequest(self, **_kwargs) -> None:
                raise RuntimeError("insert failed")

        monkeypatch.setattr(router_module, "LLMRequestRepository", BrokenRepo)
        session = _fake_session()
        llm = FakeLLM(responses=[LLMResponse(text=_valid_json("direct_answer"))])
        output, _scope = await route_query(
            RouterInput(query="What was revenue?"),
            llm_router=FakeRouter(llm),
            session=cast(AsyncSession, session),
            parent_request_id=uuid4(),
            conversation_id=uuid4(),
        )
        assert output.route == "direct_answer"
        session.rollback.assert_not_awaited()
        session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_recovers_on_retry_after_bad_first_attempt(self) -> None:
        llm = FakeLLM(responses=[LLMResponse(text="garbage"), LLMResponse(text=_valid_json())])
        router = FakeRouter(llm)
        output, scope = await route_query(RouterInput(query="What was revenue?"), llm_router=router)
        assert output.route == "retrieval"
        assert len(llm.calls) == 2
