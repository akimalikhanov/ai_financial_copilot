from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Literal, Optional, Sequence

from .base_adapter import (
    ChatMessage,
    ChatRequest,
    LLMAdapter,
    LLMResponse,
    LLMStreamChunk,
)


@dataclass(frozen=True)
class OpenAIChatRequest(ChatRequest):
    reasoning_effort: Optional[
        Literal["none", "minimal", "low", "medium", "high"]
    ] = None
    verbosity: Optional[Literal["high", "medium", "low"]] = None


class OpenAIAdapter(LLMAdapter):
    """Uses openai-python AsyncOpenAI + Chat Completions (non-streaming)."""

    def __init__(
        self,
        *,
        default_model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,  # useful for OpenAI-compatible servers
    ):
        super().__init__(default_model=default_model)

        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    @staticmethod
    def _is_gpt_5_model(model: str) -> bool:
        """Check if model is a GPT-5 model (supports reasoning_effort/verbosity, but not temperature)."""
        return model.startswith("gpt-5")

    def _build_request(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[
            Literal["none", "minimal", "low", "medium", "high"]
        ] = None,
        verbosity: Optional[Literal["high", "medium", "low"]] = None,
    ) -> OpenAIChatRequest:
        return OpenAIChatRequest(
            messages=messages,
            model=model or self.default_model,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
        )

    def _build_kwargs(self, req: ChatRequest) -> dict[str, object]:
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        is_gpt_5 = self._is_gpt_5_model(req.model)

        kwargs: dict[str, object] = {
            "model": req.model,
            "messages": messages,
        }
        # GPT-5 models don't accept temperature
        if req.temperature is not None and not is_gpt_5:
            kwargs["temperature"] = req.temperature
        if req.max_tokens is not None:
            kwargs["max_tokens"] = req.max_tokens
        
        # reasoning_effort and verbosity are only supported by GPT-5 models
        if isinstance(req, OpenAIChatRequest) and is_gpt_5:
            if req.reasoning_effort is not None:
                kwargs["reasoning_effort"] = req.reasoning_effort
            if req.verbosity is not None:
                kwargs["verbosity"] = req.verbosity
        return kwargs

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[
            Literal["none", "minimal", "low", "medium", "high"]
        ] = None,
        verbosity: Optional[Literal["high", "medium", "low"]] = None,
    ) -> LLMResponse:
        req = self._build_request(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
        )
        return await self._complete(req)

    def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[
            Literal["none", "minimal", "low", "medium", "high"]
        ] = None,
        verbosity: Optional[Literal["high", "medium", "low"]] = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        req = self._build_request(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
        )
        return self._stream(req)

    async def _complete(self, req: ChatRequest) -> LLMResponse:
        kwargs = self._build_kwargs(req)

        resp = await self._client.chat.completions.create(**kwargs)

        text = (resp.choices[0].message.content or "").strip()
        return LLMResponse(text=text, raw=resp)

    async def _stream(self, req: ChatRequest) -> AsyncIterator[LLMStreamChunk]:
        kwargs = self._build_kwargs(req)

        stream = await self._client.chat.completions.create(**kwargs, stream=True)

        # parts: list[str] = []
        async for chunk in stream:
            delta = chunk.choices[0].delta
            text = getattr(delta, "content", None) or ""
            if text:
                # parts.append(text)
                yield LLMStreamChunk(text=text, raw=chunk)

        # full_text = "".join(parts).strip()
        # yield LLMStreamChunk(text=full_text, is_final=True)
