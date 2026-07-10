"""
Unit tests for LLM router (robust + behavior-focused).

Key improvements vs previous version:
- No checks against private adapter internals (like adapter._client.base_url)
- Covers important behavior gaps:
  - _merge_params ignores None overrides
  - vLLM missing host/port errors
  - RoutedLLM.complete/stream/run correctly merges + dispatches
  - Router builds models + applies global + per-model param overrides
  - Router skips invalid model entries (missing id/provider)

NOTE:
- These tests monkeypatch OpenAIAdapter/GeminiAdapter to avoid real network/client deps.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

import src.services.llm_router as llm_router_mod
from src.services.llm_adapters.base_adapter import ChatMessage, LLMResponse, Role
from src.services.llm_router import (
    LLMRouter,
    RoutedLLM,
    _build_adapter,
    _merge_params,
    _normalize_base_url,
)
from src.services.llm_runtime.exceptions import LLMNotFoundError, LLMServerError


# -------------------------
# Test doubles (Fakes)
# -------------------------
class DummyOpenAIAdapter:
    """Test double for OpenAIAdapter (records init args)."""

    def __init__(
        self,
        default_model: str,
        base_url: str | None = None,
        include_usage: bool = False,
        provider_name: str | None = None,
    ):
        self.default_model = default_model
        self.base_url = base_url
        self.include_usage = include_usage
        self.provider_name = provider_name

    async def complete(self, messages, **params):
        return {"type": "complete", "messages": messages, "params": params}

    def stream(self, messages, **params) -> AsyncIterator[dict]:
        async def _gen():
            yield {"type": "stream_chunk", "messages": messages, "params": params}
            yield {"type": "stream_done"}

        return _gen()


class DummyGeminiAdapter:
    """Test double for GeminiAdapter (records init args)."""

    def __init__(self, default_model: str):
        self.default_model = default_model

    async def complete(self, messages, **params):
        return {"type": "complete", "messages": messages, "params": params}

    def stream(self, messages, **params) -> AsyncIterator[dict]:
        async def _gen():
            yield {"type": "stream_chunk", "messages": messages, "params": params}
            yield {"type": "stream_done"}

        return _gen()


class RecordingAdapter:
    """
    A tiny adapter that records the last call.
    Use it to validate parameter merging + dispatch behavior.
    """

    def __init__(self):
        self.last_complete: dict[str, Any] | None = None
        self.last_stream: dict[str, Any] | None = None

    async def complete(self, messages, **params):
        self.last_complete = {"messages": messages, "params": params}
        return LLMResponse(text="ok")

    def stream(self, messages, **params):
        self.last_stream = {"messages": messages, "params": params}

        async def _gen():
            yield {"chunk": 1}
            yield {"chunk": 2}

        return _gen()


@pytest.fixture(autouse=True)
def patch_adapters(monkeypatch: pytest.MonkeyPatch):
    """
    Make all router tests pure unit tests:
    replace real OpenAI/Gemini adapters with in-memory doubles.
    """
    monkeypatch.setattr(llm_router_mod, "OpenAIAdapter", DummyOpenAIAdapter)
    monkeypatch.setattr(llm_router_mod, "GeminiAdapter", DummyGeminiAdapter)


# -------------------------
# _merge_params
# -------------------------
class TestMergeParams:
    def test_merge_params_basic(self):
        defaults = {"temperature": 0.2, "max_tokens": 2000}
        overrides = {"temperature": 0.7}
        result = _merge_params(defaults, overrides)
        assert result == {"temperature": 0.7, "max_tokens": 2000}

    def test_merge_params_empty_defaults(self):
        result = _merge_params({}, {"temperature": 0.5})
        assert result == {"temperature": 0.5}

    def test_merge_params_empty_overrides(self):
        result = _merge_params({"temperature": 0.2}, {})
        assert result == {"temperature": 0.2}

    def test_merge_params_adds_new_keys(self):
        defaults = {"temperature": 0.2}
        overrides = {"max_tokens": 1000, "seed": 42}
        result = _merge_params(defaults, overrides)
        assert result == {"temperature": 0.2, "max_tokens": 1000, "seed": 42}

    def test_merge_params_ignores_none_overrides(self):
        defaults = {"temperature": 0.2, "max_tokens": 999}
        overrides = {"temperature": None, "max_tokens": 123}
        result = _merge_params(defaults, overrides)
        # temperature=None should NOT clobber defaults
        assert result == {"temperature": 0.2, "max_tokens": 123}


# -------------------------
# _normalize_base_url
# -------------------------
class TestNormalizeBaseUrl:
    def test_basic(self):
        assert _normalize_base_url("localhost", "8002", "/v1") == "http://localhost:8002/v1"

    def test_with_scheme(self):
        assert _normalize_base_url("http://localhost", "8002", "/v1") == "http://localhost:8002/v1"
        assert (
            _normalize_base_url("https://api.example.com", "443", "/v1")
            == "https://api.example.com:443/v1"
        )

    def test_no_base_path(self):
        assert _normalize_base_url("localhost", "8002") == "http://localhost:8002"

    def test_adds_leading_slash(self):
        assert _normalize_base_url("localhost", "8002", "v1") == "http://localhost:8002/v1"


# -------------------------
# _build_adapter
# -------------------------
class TestBuildAdapter:
    def test_build_openai_adapter(self):
        cfg = {"model_name": "gpt-4"}
        adapter = _build_adapter("openai", cfg)
        assert isinstance(adapter, DummyOpenAIAdapter)
        assert adapter.default_model == "gpt-4"
        assert adapter.base_url is None
        assert adapter.include_usage is False

    def test_build_gemini_adapter(self):
        cfg = {"model_name": "gemini-pro"}
        adapter = _build_adapter("google", cfg)
        assert isinstance(adapter, DummyGeminiAdapter)
        assert adapter.default_model == "gemini-pro"

    def test_build_vllm_adapter_builds_expected_base_url_and_flags(self):
        cfg = {
            "model_path": "llama-3",
            "server": {"host": "localhost", "port": "8000"},
            "include_usage": True,
        }
        adapter = _build_adapter("vllm", cfg)
        assert isinstance(adapter, DummyOpenAIAdapter)
        assert adapter.default_model == "llama-3"

        # Base URL should be produced by our own normalization function (stable check)
        assert adapter.base_url == _normalize_base_url("localhost", "8000", "/v1")
        assert adapter.include_usage is True

    def test_vllm_default_include_usage_false(self):
        cfg = {"model_path": "llama-3", "server": {"host": "localhost", "port": "8000"}}
        adapter = _build_adapter("vllm", cfg)
        assert isinstance(adapter, DummyOpenAIAdapter)
        assert adapter.include_usage is False

    def test_openai_missing_model_name(self):
        with pytest.raises(LLMServerError, match="missing model_name"):
            _build_adapter("openai", {})

    def test_google_missing_model_name(self):
        with pytest.raises(LLMServerError, match="missing model_name"):
            _build_adapter("google", {})

    def test_vllm_missing_model_path(self):
        with pytest.raises(LLMServerError, match="missing model_path"):
            _build_adapter("vllm", {"server": {"host": "localhost", "port": "8002"}})

    def test_vllm_missing_host_or_port(self):
        with pytest.raises(LLMServerError, match="missing host/port"):
            _build_adapter("vllm", {"model_path": "llama-3", "server": {"port": "8000"}})

        with pytest.raises(LLMServerError, match="missing host/port"):
            _build_adapter("vllm", {"model_path": "llama-3", "server": {"host": "localhost"}})

    def test_unsupported_provider(self):
        with pytest.raises(LLMServerError, match="Unsupported provider"):
            _build_adapter("anthropic", {})


# -------------------------
# RoutedLLM behavior tests
# -------------------------
class TestRoutedLLM:
    @pytest.mark.asyncio
    async def test_complete_merges_params_and_calls_adapter(self):
        adapter = RecordingAdapter()
        routed = RoutedLLM(
            adapter=adapter,  # type: ignore
            provider="fake",
            model_id="m1",
            default_params={"temperature": 0.2, "max_tokens": 100},
            default_stream=False,
            capabilities={},
        )

        messages = [ChatMessage(role=Role.user, content="hi")]
        await routed.complete(messages, temperature=0.9, seed=7)

        assert adapter.last_complete is not None
        assert adapter.last_complete["messages"] == messages
        assert adapter.last_complete["params"] == {"temperature": 0.9, "max_tokens": 100, "seed": 7}

    @pytest.mark.asyncio
    async def test_complete_does_not_clobber_defaults_with_none(self):
        adapter = RecordingAdapter()
        routed = RoutedLLM(
            adapter=adapter,  # type: ignore
            provider="fake",
            model_id="m1",
            default_params={"temperature": 0.2, "max_tokens": 100},
            default_stream=False,
            capabilities={},
        )

        messages = [ChatMessage(role=Role.user, content="hi")]
        await routed.complete(messages, temperature=None, max_tokens=777)

        assert adapter.last_complete is not None
        assert adapter.last_complete["params"] == {"temperature": 0.2, "max_tokens": 777}

    @pytest.mark.asyncio
    async def test_stream_merges_params_and_calls_adapter(self):
        adapter = RecordingAdapter()
        routed = RoutedLLM(
            adapter=adapter,  # type: ignore
            provider="fake",
            model_id="m1",
            default_params={"temperature": 0.2},
            default_stream=True,
            capabilities={},
        )

        messages = [ChatMessage(role=Role.user, content="hi")]
        stream_iter = routed.stream(messages, temperature=0.8)

        assert adapter.last_stream is not None
        assert adapter.last_stream["messages"] == messages
        assert adapter.last_stream["params"] == {"temperature": 0.8}

        # Consume stream to ensure it's an async iterator
        chunks = []
        async for c in stream_iter:
            chunks.append(c)
        assert len(chunks) == 2

    def test_run_chooses_stream_by_default_stream_true(self):
        adapter = RecordingAdapter()
        routed = RoutedLLM(
            adapter=adapter,  # type: ignore
            provider="fake",
            model_id="m1",
            default_params={},
            default_stream=True,
            capabilities={},
        )

        messages = [ChatMessage(role=Role.user, content="hi")]
        result = routed.run(messages)  # should stream
        assert hasattr(result, "__aiter__")  # async iterator

    @pytest.mark.asyncio
    async def test_run_chooses_complete_by_default_stream_false(self):
        adapter = RecordingAdapter()
        routed = RoutedLLM(
            adapter=adapter,  # type: ignore
            provider="fake",
            model_id="m1",
            default_params={},
            default_stream=False,
            capabilities={},
        )

        messages = [ChatMessage(role=Role.user, content="hi")]
        result = routed.run(messages)  # should complete
        assert asyncio.iscoroutine(result)
        # Await the coroutine to avoid the warning
        await result

    def test_run_explicit_stream_overrides_default(self):
        adapter = RecordingAdapter()
        routed = RoutedLLM(
            adapter=adapter,  # type: ignore
            provider="fake",
            model_id="m1",
            default_params={},
            default_stream=False,  # default is complete
            capabilities={},
        )

        messages = [ChatMessage(role=Role.user, content="hi")]
        result = routed.run(messages, stream=True)  # override to stream
        assert hasattr(result, "__aiter__")


# -------------------------
# LLMRouter behavior tests
# -------------------------
class TestLLMRouter:
    @pytest.fixture
    def router_config(self):
        return {
            "defaults": {"stream": True, "params": {"temperature": 0.5, "max_tokens": 999}},
            "models": [
                {
                    "id": "gpt-4",
                    "provider": "openai",
                    "model_name": "gpt-4",
                    "params_override": {"temperature": 1.0},  # overrides global temperature
                    "capabilities": {"vision": True},
                },
                {"id": "gemini", "provider": "google", "model_name": "gemini-pro"},
                # Invalid entries should be skipped
                {"id": None, "provider": "openai", "model_name": "x"},
                {"id": "no-provider", "provider": None, "model_name": "x"},
            ],
        }

    def test_router_init_and_list_models(self, router_config):
        router = LLMRouter(router_config)
        assert router.list_models() == ["gemini", "gpt-4"]

    def test_router_get_model_and_defaults(self, router_config):
        router = LLMRouter(router_config)
        model = router.get("gpt-4")

        assert model.model_id == "gpt-4"
        assert model.provider == "openai"
        assert model.default_stream is True

        # params = global defaults + params_override
        assert model.default_params["temperature"] == 1.0
        assert model.default_params["max_tokens"] == 999

        # capabilities passed through
        assert model.capabilities == {"vision": True}

    def test_router_get_unknown_model_raises(self, router_config):
        router = LLMRouter(router_config)
        with pytest.raises(LLMNotFoundError, match="Unknown model_id"):
            router.get("unknown")

    def test_router_builds_adapter_lazily_on_first_get(self, monkeypatch, router_config):
        calls = []

        def fake_build_adapter(provider: str, cfg: dict[str, Any]):
            calls.append((provider, cfg.get("id")))
            # return any adapter-like object
            return DummyOpenAIAdapter(default_model="x")

        monkeypatch.setattr(llm_router_mod, "_build_adapter", fake_build_adapter)

        router = LLMRouter(router_config)
        # Construction must not build any adapter — an unconfigured provider (e.g. a
        # Gemini model with no API key) should not fail router init.
        assert calls == []
        assert router.list_models() == ["gemini", "gpt-4"]

        # First get() builds exactly that model's adapter (invalid entries skipped).
        router.get("gpt-4")
        assert ("openai", "gpt-4") in calls
        assert ("google", "gemini") not in calls
        assert len(calls) == 1

        # Second get() of the same model is cached — no rebuild.
        router.get("gpt-4")
        assert len(calls) == 1

        # Getting the other model builds only that one.
        router.get("gemini")
        assert ("google", "gemini") in calls
        assert len(calls) == 2
