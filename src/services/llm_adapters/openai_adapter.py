from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

from src.services.llm_runtime.exception_mapper import map_openai_error
from src.utils.llm_utils import (
    calc_cost_openai,
    compute_tps,
    elapsed_ms,
    get_pricing_for_model,
    now_ms,
)

from .base_adapter import (
    AssistantTurnResult,
    ChatMessage,
    ChatRequest,
    LLMAdapter,
    LLMResponse,
    LLMResponseStats,
    LLMStreamChunk,
    ToolCallRef,
)


@dataclass(frozen=True)
class OpenAIChatRequest(ChatRequest):
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high"] | None = None
    verbosity: Literal["high", "medium", "low"] | None = None


class OpenAIAdapter(LLMAdapter):
    """Uses openai-python AsyncOpenAI + Chat Completions (non-streaming)."""

    provider_name: str = "openai"

    def __init__(
        self,
        *,
        default_model: str,
        api_key: str | None = None,
        base_url: str | None = None,  # useful for OpenAI-compatible servers
        include_usage: bool = True,
        provider_name: str | None = None,  # override for vLLM or other compatible servers
    ):
        super().__init__(default_model=default_model)
        self.include_usage = include_usage
        if provider_name is not None:
            self.provider_name = provider_name

        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()

    @staticmethod
    def _is_gpt_5_model(model: str) -> bool:
        """Check if model is a GPT-5 model (supports reasoning_effort/verbosity, but not temperature)."""
        return model.startswith("gpt-5")

    @staticmethod
    def _build_stats_from_usage(
        usage: object | None,
        *,
        model: str,
        latency_ms: float,
        ttft_ms: float | None = None,
    ) -> LLMResponseStats:
        input_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        output_tokens = getattr(usage, "completion_tokens", None) if usage else None
        total_tokens = getattr(usage, "total_tokens", None) if usage else None

        reasoning_tokens = None
        cached_input_tokens = None
        if usage is not None:
            completion_details = getattr(usage, "completion_tokens_details", None)
            if completion_details is not None:
                reasoning_tokens = getattr(completion_details, "reasoning_tokens", None)
            prompt_details = getattr(usage, "prompt_tokens_details", None)
            if prompt_details is not None:
                cached_input_tokens = getattr(prompt_details, "cached_tokens", None)

        stats = LLMResponseStats(
            input_tokens=input_tokens,
            cached_input_tokens=cached_input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            tps=compute_tps(
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                ttft_ms=ttft_ms,
            ),
        )
        pricing = get_pricing_for_model("openai", model)
        if pricing is not None:
            cost = calc_cost_openai(stats, pricing)
            if cost is not None:
                stats = replace(stats, cost_usd=cost)
        return stats

    def _build_request(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: Literal["none", "minimal", "low", "medium", "high"] | None = None,
        verbosity: Literal["high", "medium", "low"] | None = None,
        **kwargs: Any,
    ) -> OpenAIChatRequest:
        return OpenAIChatRequest(
            messages=tuple(messages),
            model=model or self.default_model,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            extra_params=kwargs,
        )

    @staticmethod
    def _serialize_msg(m: ChatMessage) -> dict[str, Any]:
        d: dict[str, Any] = {"role": m.role}
        d["content"] = m.content if m.content is not None else ""
        if m.name:
            d["name"] = m.name
        if m.tool_call_id:
            d["tool_call_id"] = m.tool_call_id
        if m.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in m.tool_calls
            ]
        return d

    def _build_kwargs(self, req: ChatRequest) -> dict[str, Any]:
        messages = [self._serialize_msg(m) for m in req.messages]
        is_gpt_5 = self._is_gpt_5_model(req.model)

        kwargs: dict[str, Any] = {
            "model": req.model,
            "messages": messages,
        }
        # GPT-5 models don't accept temperature
        if req.temperature is not None and not is_gpt_5:
            kwargs["temperature"] = req.temperature
        if req.max_tokens is not None:
            if is_gpt_5:
                kwargs["max_completion_tokens"] = req.max_tokens
            else:
                kwargs["max_tokens"] = req.max_tokens

        # reasoning_effort and verbosity are only supported by GPT-5 models
        if isinstance(req, OpenAIChatRequest) and is_gpt_5:
            if req.reasoning_effort is not None:
                kwargs["reasoning_effort"] = req.reasoning_effort
            if req.verbosity is not None:
                kwargs["verbosity"] = req.verbosity

        # Merge extra params from ChatRequest.
        # enable_thinking is Qwen3-specific: must be nested under extra_body.chat_template_kwargs.
        if req.extra_params:
            extra = dict(req.extra_params)
            if "enable_thinking" in extra:
                is_qwen = req.model.lower().startswith("qwen")
                if is_qwen:
                    kwargs.setdefault("extra_body", {}).setdefault("chat_template_kwargs", {})[
                        "enable_thinking"
                    ] = extra.pop("enable_thinking")
                else:
                    extra.pop("enable_thinking")  # silently drop for non-Qwen models
            kwargs.update(extra)

        return kwargs

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: Literal["none", "minimal", "low", "medium", "high"] | None = None,
        verbosity: Literal["high", "medium", "low"] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        req = self._build_request(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            **kwargs,
        )
        return await self._complete(req)

    def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: Literal["none", "minimal", "low", "medium", "high"] | None = None,
        verbosity: Literal["high", "medium", "low"] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[LLMStreamChunk, None]:
        req = self._build_request(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            **kwargs,
        )
        return self._stream(req)

    async def _complete(self, req: ChatRequest) -> LLMResponse:
        kwargs = self._build_kwargs(req)

        start_ms = now_ms()
        try:
            resp = await self._client.chat.completions.create(**cast(Any, kwargs))
        except Exception as e:
            raise map_openai_error(e, provider=self.provider_name, model=req.model) from e
        latency_ms = elapsed_ms(start_ms)

        text = (resp.choices[0].message.content or "").strip()
        stats = self._build_stats_from_usage(
            getattr(resp, "usage", None),
            model=req.model,
            latency_ms=latency_ms,
        )

        return LLMResponse(text=text, raw=resp, stats=stats)

    async def complete_with_tools(
        self,
        messages: Sequence[ChatMessage],
        tools: list[dict[str, Any]],
        **kwargs: Any,
    ) -> AssistantTurnResult:
        req = self._build_request(messages, **kwargs)
        kw = self._build_kwargs(req)
        kw["tools"] = tools
        kw["tool_choice"] = "auto"
        start_ms = now_ms()
        try:
            resp = await self._client.chat.completions.create(**cast(Any, kw))
        except Exception as e:
            raise map_openai_error(e, provider=self.provider_name, model=req.model) from e
        latency_ms = elapsed_ms(start_ms)
        msg = resp.choices[0].message
        tool_calls = [
            ToolCallRef(id=tc.id, name=tc.function.name, arguments=tc.function.arguments)
            for tc in (msg.tool_calls or [])
        ]
        stats = self._build_stats_from_usage(
            getattr(resp, "usage", None), model=req.model, latency_ms=latency_ms
        )
        return AssistantTurnResult(text=msg.content or "", tool_calls=tool_calls, stats=stats)

    async def _stream(self, req: ChatRequest) -> AsyncGenerator[LLMStreamChunk, None]:
        kwargs = self._build_kwargs(req)
        if self.include_usage and "stream_options" not in kwargs:
            kwargs["stream_options"] = {"include_usage": True}

        start_ms = now_ms()
        first_token_ms: float | None = None

        try:
            stream = await self._client.chat.completions.create(
                **cast(Any, kwargs),
                stream=True,
            )
        except Exception as e:
            raise map_openai_error(e, provider=self.provider_name, model=req.model) from e

        try:
            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                text = getattr(delta, "content", None) or ""
                usage = getattr(chunk, "usage", None)

                # Track first token time
                if text and first_token_ms is None:
                    first_token_ms = now_ms()

                # If this is the final chunk with usage, compute stats
                if usage:
                    latency_ms = elapsed_ms(start_ms)
                    ttft_ms = elapsed_ms(start_ms, first_token_ms) if first_token_ms else None
                    stats = self._build_stats_from_usage(
                        usage,
                        model=req.model,
                        latency_ms=latency_ms,
                        ttft_ms=ttft_ms,
                    )

                    yield LLMStreamChunk(text=text, raw=chunk, is_final=True, stats=stats)
                elif text:
                    yield LLMStreamChunk(text=text, raw=chunk, is_final=False)
        except Exception as e:
            raise map_openai_error(e, provider=self.provider_name, model=req.model) from e
