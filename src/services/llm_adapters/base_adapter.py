from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, Literal, Optional, Sequence

Role = Literal["system", "developer", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


@dataclass(frozen=True)
class ChatRequest:
    messages: Sequence[ChatMessage]
    model: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


@dataclass(frozen=True)
class LLMResponse:
    text: str
    raw: Any = None  # provider-native response object (optional)


@dataclass(frozen=True)
class LLMStreamChunk:
    text: str
    raw: Any = None  # provider-native chunk object (optional)
    is_final: bool = False


class LLMAdapter(ABC):
    """Provider-agnostic non-streaming interface."""

    def __init__(self, *, default_model: str):
        self.default_model = default_model

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        req = ChatRequest(
            messages=messages,
            model=model or self.default_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return await self._complete(req)

    def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        req = ChatRequest(
            messages=messages,
            model=model or self.default_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return self._stream(req)

    @abstractmethod
    async def _complete(self, req: ChatRequest) -> LLMResponse:
        raise NotImplementedError

    @abstractmethod
    async def _stream(self, req: ChatRequest) -> AsyncIterator[LLMStreamChunk]:
        raise NotImplementedError
