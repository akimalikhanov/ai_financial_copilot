from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any, cast

from src.services.llm_runtime.exception_mapper import map_google_error
from src.utils.llm_utils import (
    calc_cost_google,
    compute_tps,
    elapsed_ms,
    get_pricing_for_model,
    now_ms,
)

from .base_adapter import ChatRequest, LLMAdapter, LLMResponse, LLMResponseStats, LLMStreamChunk


class GeminiAdapter(LLMAdapter):
    """Uses google-genai (Gemini Developer API)."""

    provider_name: str = "google"

    def __init__(
        self,
        *,
        default_model: str,
        api_key: str | None = None,  # if None, google-genai reads GEMINI_API_KEY
    ):
        super().__init__(default_model=default_model)

        from google import genai

        # If api_key is None, genai.Client() will read GEMINI_API_KEY from env.
        self._client = genai.Client(api_key=api_key) if api_key else genai.Client()

    def _build_contents_and_config(self, req: ChatRequest):
        from google.genai import types

        # Convert "system/developer" messages into a single system instruction
        sys_parts: list[str] = []
        contents: list[types.Content] = []

        for m in req.messages:
            if m.role in ("system", "developer"):
                sys_parts.append(m.content)
                continue

            # Gemini uses "user" and "model" roles for chat history
            role = "user" if m.role == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=m.content)]))

        config_kwargs: dict[str, Any] = {}
        if sys_parts:
            config_kwargs["system_instruction"] = "\n".join(sys_parts)
        if req.temperature is not None:
            config_kwargs["temperature"] = req.temperature
        if req.max_tokens is not None:
            config_kwargs["max_output_tokens"] = req.max_tokens

        config = (
            types.GenerateContentConfig(**config_kwargs)
            if config_kwargs
            else types.GenerateContentConfig()
        )
        return contents, config

    @staticmethod
    def _build_stats_from_usage_metadata(
        usage_metadata: object | None,
        *,
        model: str,
        latency_ms: float,
        ttft_ms: float | None = None,
    ) -> LLMResponseStats:
        input_tokens = (
            getattr(usage_metadata, "prompt_token_count", None) if usage_metadata else None
        )
        output_tokens = (
            getattr(usage_metadata, "candidates_token_count", None) if usage_metadata else None
        )
        total_tokens = (
            getattr(usage_metadata, "total_token_count", None) if usage_metadata else None
        )
        reasoning_tokens = (
            getattr(usage_metadata, "thoughts_token_count", None) if usage_metadata else None
        )
        cached_input_tokens = (
            getattr(usage_metadata, "cached_content_token_count", None) if usage_metadata else None
        )

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
        pricing = get_pricing_for_model("google", model)
        if pricing is not None:
            cost = calc_cost_google(stats, pricing)
            if cost is not None:
                stats = replace(stats, cost_usd=cost)
        return stats

    async def _complete(self, req: ChatRequest) -> LLMResponse:
        contents, config = self._build_contents_and_config(req)

        start_ms = now_ms()
        try:
            resp = await self._client.aio.models.generate_content(
                model=req.model,
                contents=cast(Any, contents),
                config=config,
            )
        except Exception as e:
            raise map_google_error(e, provider=self.provider_name, model=req.model) from e
        latency_ms = elapsed_ms(start_ms)

        text = (resp.text or "").strip()

        stats = self._build_stats_from_usage_metadata(
            getattr(resp, "usage_metadata", None),
            model=req.model,
            latency_ms=latency_ms,
        )

        return LLMResponse(text=text, raw=resp, stats=stats)

    async def _stream(self, req: ChatRequest) -> AsyncIterator[LLMStreamChunk]:
        from inspect import isawaitable

        contents, config = self._build_contents_and_config(req)

        start_ms = now_ms()
        first_token_ms: float | None = None
        last_usage_metadata = None
        last_chunk = None

        try:
            stream = self._client.aio.models.generate_content_stream(
                model=req.model,
                contents=cast(Any, contents),
                config=config,
            )
            if isawaitable(stream):
                stream = await stream
        except Exception as e:
            raise map_google_error(e, provider=self.provider_name, model=req.model) from e

        try:
            async for chunk in stream:
                last_chunk = chunk
                text = getattr(chunk, "text", None) or ""
                if text and first_token_ms is None:
                    first_token_ms = now_ms()

                usage_metadata = getattr(chunk, "usage_metadata", None)
                if usage_metadata is not None:
                    last_usage_metadata = usage_metadata

                if text:
                    yield LLMStreamChunk(text=text, raw=chunk)
        except Exception as e:
            raise map_google_error(e, provider=self.provider_name, model=req.model) from e

        if last_usage_metadata is not None:
            latency_ms = elapsed_ms(start_ms)
            ttft_ms = elapsed_ms(start_ms, first_token_ms) if first_token_ms else None
            stats = self._build_stats_from_usage_metadata(
                last_usage_metadata,
                model=req.model,
                latency_ms=latency_ms,
                ttft_ms=ttft_ms,
            )

            yield LLMStreamChunk(text="", raw=last_chunk, is_final=True, stats=stats)
