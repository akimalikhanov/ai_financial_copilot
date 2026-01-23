from __future__ import annotations
from typing import Optional
from .base_adapter import ChatRequest, LLMAdapter, LLMResponse


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

    async def _complete(self, req: ChatRequest) -> LLMResponse:
        messages = [{"role": m.role, "content": m.content} for m in req.messages]

        resp = await self._client.chat.completions.create(
            model=req.model,
            messages=messages,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )

        text = (resp.choices[0].message.content or "").strip()
        return LLMResponse(text=text, raw=resp)
