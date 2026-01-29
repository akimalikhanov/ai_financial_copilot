# llm_router_runtime.py
from __future__ import annotations

from collections.abc import AsyncIterator, Coroutine, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from src.services.llm_adapters.base_adapter import (
    ChatMessage,
    LLMAdapter,
    LLMResponse,
    LLMStreamChunk,
)
from src.services.llm_adapters.gemini_adapter import GeminiAdapter
from src.services.llm_adapters.openai_adapter import OpenAIAdapter
from src.services.llm_runtime.exceptions import LLMNotFoundError, LLMServerError
from src.utils.config import load_models_config


def _merge_params(defaults: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(defaults)
    for k, v in overrides.items():
        # Optional: ignore None so "unset" doesn't clobber defaults
        if v is not None:
            merged[k] = v
    return merged


def _normalize_base_url(host: str, port: Any, base_path: str = "") -> str:
    host = host.strip()
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    if base_path and not base_path.startswith("/"):
        base_path = "/" + base_path
    return f"{host}:{port}{base_path}"


def _build_adapter(provider: str, model_cfg: Mapping[str, Any]) -> LLMAdapter:
    if provider == "openai":
        model_name = model_cfg.get("model_name")
        if not model_name:
            raise LLMServerError(
                "OpenAI model is missing model_name",
                provider=provider,
                status_code=500,
            )
        return OpenAIAdapter(default_model=model_name)

    if provider == "google":
        model_name = model_cfg.get("model_name")
        if not model_name:
            raise LLMServerError(
                "Google model is missing model_name",
                provider=provider,
                status_code=500,
            )
        return GeminiAdapter(default_model=model_name)

    if provider == "vllm":
        model_path = model_cfg.get("model_path")
        server = model_cfg.get("server") or {}
        if not model_path:
            raise LLMServerError(
                "vLLM model is missing model_path",
                provider=provider,
                status_code=500,
            )

        host = server.get("host")
        port = server.get("port")
        base_path = server.get("base_path") or "/v1"
        if not host or not port:
            raise LLMServerError(
                "vLLM server config missing host/port",
                provider=provider,
                status_code=500,
            )

        base_url = _normalize_base_url(host, port, base_path)

        # vLLM often doesn't support stream_options={"include_usage": True}
        # We default to False unless explicitly enabled in config
        include_usage = bool(model_cfg.get("include_usage", False))
        return OpenAIAdapter(
            default_model=model_path,
            base_url=base_url,
            include_usage=include_usage,
            provider_name="vllm",
        )

    raise LLMServerError(f"Unsupported provider: {provider!r}", provider=provider, status_code=500)


@dataclass(frozen=True)
class RoutedLLM:
    adapter: LLMAdapter
    provider: str
    model_id: str
    default_params: dict[str, Any]
    default_stream: bool
    capabilities: dict[str, Any]

    async def complete(self, messages: Sequence[ChatMessage], **params: Any) -> LLMResponse:
        merged = _merge_params(self.default_params, params)
        return await self.adapter.complete(messages=messages, **merged)

    def stream(
        self, messages: Sequence[ChatMessage], **params: Any
    ) -> AsyncIterator[LLMStreamChunk]:
        merged = _merge_params(self.default_params, params)
        return self.adapter.stream(messages=messages, **merged)

    def run(
        self,
        messages: Sequence[ChatMessage],
        stream: bool | None = None,
        **params: Any,
    ) -> Coroutine[Any, Any, LLMResponse] | AsyncIterator[LLMStreamChunk]:
        """
        Convenience method that delegates to stream() or complete() based on configuration.
        """
        should_stream = stream if stream is not None else self.default_stream
        if should_stream:
            return self.stream(messages, **params)
        else:
            return self.complete(messages, **params)


class LLMRouter:
    def __init__(self, config: Mapping[str, Any]):
        self._config = config
        self._models: dict[str, RoutedLLM] = {}

        defaults = config.get("defaults") or {}
        global_default_params = dict(defaults.get("params") or {})
        global_default_stream = bool(defaults.get("stream", False))

        for m in config.get("models") or []:
            model_id = m.get("id")
            provider = m.get("provider")
            if not model_id or not provider:
                continue

            adapter = _build_adapter(provider, m)

            params = dict(global_default_params)
            params.update(m.get("params_override") or {})

            self._models[model_id] = RoutedLLM(
                adapter=adapter,
                provider=provider,
                model_id=model_id,
                default_params=params,
                default_stream=global_default_stream,
                capabilities=dict(m.get("capabilities") or {}),
            )

    def get(self, model_id: str) -> RoutedLLM:
        try:
            return self._models[model_id]
        except KeyError:
            raise LLMNotFoundError(
                f"Unknown model_id: {model_id}",
                model=model_id,
                status_code=404,
            ) from None

    def list_models(self) -> list[str]:
        return sorted(self._models.keys())

    async def close(self) -> None:
        """Close all adapter HTTP clients for graceful shutdown."""
        for routed in self._models.values():
            await routed.adapter.close()


@lru_cache(maxsize=1)
def get_router(config_path: str | None = None) -> LLMRouter:
    cfg = load_models_config(config_path)
    return LLMRouter(cfg)
