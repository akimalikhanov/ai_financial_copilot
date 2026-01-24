from __future__ import annotations
from typing import Any, AsyncIterator, Optional
from .base_adapter import ChatRequest, LLMAdapter, LLMResponse, LLMStreamChunk


class GeminiAdapter(LLMAdapter):
    """Uses google-genai (Gemini Developer API)."""

    def __init__(
        self,
        *,
        default_model: str,
        api_key: Optional[str] = None,  # if None, google-genai reads GEMINI_API_KEY
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
            contents.append(
                types.Content(role=role, parts=[types.Part.from_text(text=m.content)])
            )

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

    async def _complete(self, req: ChatRequest) -> LLMResponse:
        contents, config = self._build_contents_and_config(req)

        resp = await self._client.aio.models.generate_content(
            model=req.model,
            contents=contents if contents else "Hello",
            config=config,
        )

        text = (resp.text or "").strip()
        return LLMResponse(text=text, raw=resp)

    async def _stream(self, req: ChatRequest) -> AsyncIterator[LLMStreamChunk]:
        from inspect import isawaitable

        contents, config = self._build_contents_and_config(req)

        stream = self._client.aio.models.generate_content_stream(
            model=req.model,
            contents=contents if contents else "Hello",
            config=config,
        )
        if isawaitable(stream):
            stream = await stream

        async for chunk in stream:
            text = (getattr(chunk, "text", None) or "")
            if text:
                yield LLMStreamChunk(text=text, raw=chunk)
