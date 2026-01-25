"""Unit tests for LLM router."""

from __future__ import annotations

from typing import cast

import pytest

from src.services.llm_adapters.gemini_adapter import GeminiAdapter
from src.services.llm_adapters.openai_adapter import OpenAIAdapter
from src.services.llm_router import (
    LLMRouter,
    LLMRouterError,
    _build_adapter,
    _merge_params,
    _normalize_base_url,
)


class TestMergeParams:
    """Test parameter merging logic."""

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


class TestNormalizeBaseUrl:
    """Test base URL normalization."""

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


class TestBuildAdapter:
    """Test _build_adapter logic."""

    def test_build_openai_adapter(self):
        cfg = {"model_name": "gpt-4"}
        adapter = _build_adapter("openai", cfg)
        assert isinstance(adapter, OpenAIAdapter)
        assert adapter.default_model == "gpt-4"

    def test_build_gemini_adapter(self):
        cfg = {"model_name": "gemini-pro"}
        adapter = _build_adapter("google", cfg)
        assert isinstance(adapter, GeminiAdapter)
        assert adapter.default_model == "gemini-pro"

    def test_build_vllm_adapter(self):
        cfg = {
            "model_path": "llama-3",
            "server": {"host": "localhost", "port": "8000"},
            "include_usage": True,
        }
        adapter = _build_adapter("vllm", cfg)
        assert isinstance(adapter, OpenAIAdapter)
        adapter = cast(OpenAIAdapter, adapter)
        assert adapter.default_model == "llama-3"
        assert adapter._client.base_url == "http://localhost:8000/v1/"
        assert adapter.include_usage is True

    def test_vllm_default_include_usage(self):
        cfg = {"model_path": "llama-3", "server": {"host": "localhost", "port": "8000"}}
        adapter = _build_adapter("vllm", cfg)
        assert isinstance(adapter, OpenAIAdapter)
        adapter = cast(OpenAIAdapter, adapter)
        assert adapter.include_usage is False

    def test_openai_missing_model_name(self):
        with pytest.raises(LLMRouterError, match="missing model_name"):
            _build_adapter("openai", {})

    def test_google_missing_model_name(self):
        with pytest.raises(LLMRouterError, match="missing model_name"):
            _build_adapter("google", {})

    def test_vllm_missing_model_path(self):
        with pytest.raises(LLMRouterError, match="missing model_path"):
            _build_adapter("vllm", {"server": {"host": "localhost", "port": "8002"}})

    def test_unsupported_provider(self):
        with pytest.raises(LLMRouterError, match="Unsupported provider"):
            _build_adapter("anthropic", {})


class TestLLMRouter:
    """Test LLMRouter class."""

    @pytest.fixture
    def router_config(self):
        return {
            "defaults": {"stream": True, "params": {"temperature": 0.5}},
            "models": [
                {
                    "id": "gpt-4",
                    "provider": "openai",
                    "model_name": "gpt-4",
                    "params_override": {"temperature": 1.0},
                },
                {"id": "gemini", "provider": "google", "model_name": "gemini-pro"},
            ],
        }

    def test_router_init(self, router_config):
        router = LLMRouter(router_config)
        assert "gpt-4" in router.list_models()
        assert "gemini" in router.list_models()

    def test_router_get_model(self, router_config):
        router = LLMRouter(router_config)
        model = router.get("gpt-4")
        assert model.model_id == "gpt-4"
        assert model.default_params["temperature"] == 1.0  # Override
        assert model.default_stream is True

    def test_router_get_unknown_model(self, router_config):
        router = LLMRouter(router_config)
        with pytest.raises(LLMRouterError, match="Unknown model_id"):
            router.get("unknown")

    def test_router_get_adapter_reuse(self, router_config):
        """Ensure adapters are instantiated once per router instance."""
        router = LLMRouter(router_config)
        m1 = router.get("gpt-4")
        m2 = router.get("gpt-4")
        assert m1 is m2  # RoutedLLM objects are cached in the dict
