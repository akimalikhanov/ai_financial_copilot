# llm_router_runtime.py
from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Coroutine, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from src.observability import langfuse as _lf_mod
from src.services.llm_adapters.base_adapter import (
    AssistantTurnResult,
    ChatMessage,
    LLMAdapter,
    LLMResponse,
    LLMStreamChunk,
)
from src.services.llm_adapters.gemini_adapter import GeminiAdapter
from src.services.llm_adapters.openai_adapter import OpenAIAdapter
from src.services.llm_runtime.exceptions import LLMError, LLMNotFoundError, LLMServerError
from src.utils.config import load_models_config

logger = logging.getLogger(__name__)


def _role_str(role: Any) -> str:
    return role.value if hasattr(role, "value") else role


def _trace_content(m: ChatMessage) -> Any:
    """Text, or an OpenAI-style content parts list when the message carries images.

    The shape matters: Langfuse's media manager walks the payload for strings that look like
    base64 data URIs, uploads them to its own object storage and leaves a reference token, so
    the crop is viewable in the trace and the blob never reaches ClickHouse. Reusing the
    OpenAI serializer keeps the trace identical to the wire payload on that provider, and
    puts the data URI where Langfuse looks for it on every other one.
    """
    if not m.images:
        return m.content or ""
    parts: list[dict[str, Any]] = []
    if m.content:
        parts.append({"type": "text", "text": m.content})
    parts.extend(OpenAIAdapter._serialize_image(img) for img in m.images)
    return parts


def _trace_message(m: ChatMessage) -> dict[str, Any]:
    """Serialize a message for a Langfuse observation input, preserving tool-call
    structure so tool-calling turns are legible in the trace (not a bare content: "")."""
    out: dict[str, Any] = {"role": _role_str(m.role), "content": _trace_content(m)}
    if m.tool_call_id:
        out["tool_call_id"] = m.tool_call_id
    if m.tool_calls:
        out["tool_calls"] = [
            {"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in m.tool_calls
        ]
    return out


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

    async def complete(
        self, messages: Sequence[ChatMessage], *, _lf_name: str = "llm.complete", **params: Any
    ) -> LLMResponse:
        lf = _lf_mod.get_client()
        merged = _merge_params(self.default_params, params)
        if lf is None:
            return await self.adapter.complete(messages=messages, **merged)
        with lf.start_as_current_observation(
            as_type="generation",
            name=_lf_name,
            model=self.model_id,
            input=[_trace_message(m) for m in messages],
        ) as gen:
            response = await self.adapter.complete(messages=messages, **merged)
            update_kwargs: dict = {"output": response.text}
            if response.stats:
                s = response.stats
                update_kwargs["usage_details"] = {
                    k: v
                    for k, v in {
                        "input": s.input_tokens,
                        "output": s.output_tokens,
                        "cache_read_input_tokens": s.cached_input_tokens,
                        "total": s.total_tokens,
                    }.items()
                    if v is not None
                }
                if s.cost_usd is not None:
                    update_kwargs["cost_details"] = {"total": s.cost_usd}
            gen.update(**update_kwargs)
            return response

    async def complete_with_tools(
        self,
        messages: Sequence[ChatMessage],
        tools: list[dict],
        **params: Any,
    ) -> AssistantTurnResult:
        if not hasattr(self.adapter, "complete_with_tools"):
            raise NotImplementedError(f"{self.model_id} adapter does not support tool calling")
        lf = _lf_mod.get_client()
        merged = _merge_params(self.default_params, params)
        if lf is None:
            return await self.adapter.complete_with_tools(messages=messages, tools=tools, **merged)  # type: ignore[union-attr]
        with lf.start_as_current_observation(
            as_type="generation",
            name="llm.complete_with_tools",
            model=self.model_id,
            input=[_trace_message(m) for m in messages],
            metadata={"tools": [t["function"]["name"] for t in tools if "function" in t]},
        ) as gen:
            result = await self.adapter.complete_with_tools(
                messages=messages, tools=tools, **merged
            )  # type: ignore[union-attr]
            update_kwargs: dict = {
                "output": [
                    {"name": tc.name, "arguments": tc.arguments} for tc in (result.tool_calls or [])
                ]
                or result.text,
            }
            if result.stats:
                s = result.stats
                update_kwargs["usage_details"] = {
                    k: v
                    for k, v in {
                        "input": s.input_tokens,
                        "output": s.output_tokens,
                        "cache_read_input_tokens": s.cached_input_tokens,
                        "total": s.total_tokens,
                    }.items()
                    if v is not None
                }
                if s.cost_usd is not None:
                    update_kwargs["cost_details"] = {"total": s.cost_usd}
            gen.update(**update_kwargs)
            return result

    def stream(
        self, messages: Sequence[ChatMessage], **params: Any
    ) -> AsyncGenerator[LLMStreamChunk, None]:
        merged = _merge_params(self.default_params, params)
        return self.adapter.stream(messages=messages, **merged)

    def run(
        self,
        messages: Sequence[ChatMessage],
        stream: bool | None = None,
        **params: Any,
    ) -> Coroutine[Any, Any, LLMResponse] | AsyncGenerator[LLMStreamChunk, None]:
        """
        Convenience method that delegates to stream() or complete() based on configuration.
        """
        should_stream = stream if stream is not None else self.default_stream
        if should_stream:
            return self.stream(messages, **params)
        else:
            return self.complete(messages, **params)


class FallbackStream:
    """Streams from the first model in `chain` that responds; advances to the
    next model only if the previous one raised before yielding any content
    (so partial output already sent to the user is never duplicated/lost).

    `served` reflects whichever model actually produced the response — check
    it after iteration if the caller needs to log/persist the serving model.
    """

    def __init__(self, chain: Sequence[RoutedLLM], messages: Sequence[ChatMessage], **params: Any):
        self._chain = chain
        self._messages = messages
        self._params = params
        self.served: RoutedLLM = chain[0]

    async def __aiter__(self) -> AsyncGenerator[LLMStreamChunk, None]:
        last_err: LLMError | None = None
        for i, llm in enumerate(self._chain):
            self.served = llm
            got_chunk = False
            try:
                async for chunk in llm.stream(self._messages, **self._params):
                    got_chunk = True
                    yield chunk
                return
            except LLMError as e:
                last_err = e
                if got_chunk or i == len(self._chain) - 1:
                    raise
                logger.warning(
                    "llm_fallback",
                    extra={
                        "from_model": llm.model_id,
                        "to_model": self._chain[i + 1].model_id,
                        "error": type(e).__name__,
                    },
                )
        if last_err:
            raise last_err


class LLMRouter:
    def __init__(self, config: Mapping[str, Any]):
        self._config = config

        defaults = config.get("defaults") or {}
        self._global_default_params = dict(defaults.get("params") or {})
        self._global_default_stream = bool(defaults.get("stream", False))

        # Store raw per-model config; build the adapter lazily on first get(). This
        # keeps an unconfigured provider (e.g. a Gemini model with no GEMINI_API_KEY)
        # from failing router construction — it only fails if that model is actually
        # routed to. Adapters that don't validate credentials eagerly (OpenAIAdapter)
        # were already tolerant; Gemini's genai.Client() validates at construction.
        self._model_cfgs: dict[str, Mapping[str, Any]] = {}
        for m in config.get("models") or []:
            model_id = m.get("id")
            provider = m.get("provider")
            if not model_id or not provider:
                continue
            self._model_cfgs[model_id] = m

        self._models: dict[str, RoutedLLM] = {}

    def default_params_for(self, model_id: str) -> dict[str, Any]:
        """Merged default params (global defaults + per-model params_override) without
        building the model's adapter, so it's safe to call for unconfigured providers."""
        cfg = self._model_cfgs.get(model_id)
        if cfg is None:
            raise LLMNotFoundError(f"Unknown model_id: {model_id}")
        params = dict(self._global_default_params)
        params.update(cfg.get("params_override") or {})
        return params

    def _build_routed(self, model_id: str, m: Mapping[str, Any]) -> RoutedLLM:
        provider = m["provider"]
        adapter = _build_adapter(provider, m)

        params = self.default_params_for(model_id)

        return RoutedLLM(
            adapter=adapter,
            provider=provider,
            model_id=model_id,
            default_params=params,
            default_stream=self._global_default_stream,
            capabilities=dict(m.get("capabilities") or {}),
        )

    def get(self, model_id: str) -> RoutedLLM:
        routed = self._models.get(model_id)
        if routed is not None:
            return routed

        cfg = self._model_cfgs.get(model_id)
        if cfg is None:
            raise LLMNotFoundError(
                f"Unknown model_id: {model_id}",
                model=model_id,
                status_code=404,
            ) from None

        routed = self._build_routed(model_id, cfg)
        self._models[model_id] = routed
        return routed

    def get_with_fallback(self, model_id: str) -> list[RoutedLLM]:
        """Primary model followed by its configured `fallback_model`, if any.
        Always at least length 1."""
        chain = [self.get(model_id)]
        fallback_id = self._model_cfgs[model_id].get("fallback_model")
        if fallback_id and fallback_id in self._model_cfgs:
            chain.append(self.get(fallback_id))
        return chain

    def list_models(self) -> list[str]:
        return sorted(self._model_cfgs.keys())

    async def close(self) -> None:
        """Close all instantiated adapter HTTP clients for graceful shutdown."""
        for routed in self._models.values():
            await routed.adapter.close()


@lru_cache(maxsize=1)
def get_router(config_path: str | None = None) -> LLMRouter:
    cfg = load_models_config(config_path)
    return LLMRouter(cfg)
