"""Unit tests for route_query (DI via llm_router param)."""

from __future__ import annotations

import json
from typing import cast

import pytest

from src.schemas.query_router import RouterInput
from src.services.llm_adapters.base_adapter import LLMResponse
from src.services.llm_router import LLMRouter
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
    async def test_recovers_on_retry_after_bad_first_attempt(self) -> None:
        llm = FakeLLM(responses=[LLMResponse(text="garbage"), LLMResponse(text=_valid_json())])
        router = FakeRouter(llm)
        output, scope = await route_query(RouterInput(query="What was revenue?"), llm_router=router)
        assert output.route == "retrieval"
        assert len(llm.calls) == 2
