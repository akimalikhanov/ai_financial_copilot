from __future__ import annotations
from typing import Optional
from .base_adapter import ChatRequest, LLMAdapter, LLMResponse


class GeminiAdapter(LLMAdapter):
    """Uses google-genai (Gemini Developer API) (non-streaming)."""

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

    async def _complete(self, req: ChatRequest) -> LLMResponse:
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

        config = types.GenerateContentConfig(
            system_instruction="\n".join(sys_parts) if sys_parts else None,
            temperature=req.temperature,
            max_output_tokens=req.max_tokens,
        )

        resp = await self._client.aio.models.generate_content(
            model=req.model,
            contents=contents if contents else "Hello",
            config=config,
        )

        text = (resp.text or "").strip()
        return LLMResponse(text=text, raw=resp)
