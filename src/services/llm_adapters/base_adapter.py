from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    system = "system"
    developer = "developer"
    user = "user"
    assistant = "assistant"
    tool = "tool"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Role
    content: str
    name: str | None = None  # tool name (or function name)
    tool_call_id: str | None = None  # ties tool result to assistant tool call


@dataclass(frozen=True, slots=True)
class ChatRequest:
    messages: tuple[ChatMessage, ...]
    model: str
    temperature: float | None = None
    max_tokens: int | None = None
    extra_params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LLMResponseStats:
    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: float | None = None
    ttft_ms: float | None = None
    tps: float | None = None
    cost_usd: float | None = None


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    raw: Any = None  # provider-native response object (optional)
    stats: LLMResponseStats | None = None


@dataclass(frozen=True, slots=True)
class LLMStreamChunk:
    text: str
    raw: Any = None  # provider-native chunk object (optional)
    is_final: bool = False
    stats: LLMResponseStats | None = None


class LLMAdapter(ABC):
    """Provider-agnostic non-streaming interface."""

    # Subclasses should override this to identify the provider for error mapping
    provider_name: str = "unknown"

    def __init__(self, *, default_model: str):
        self.default_model = default_model

    async def close(self) -> None:
        """Clean up resources (HTTP clients, connections). Override in subclasses."""
        pass

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        req = ChatRequest(
            messages=tuple(messages),
            model=model or self.default_model,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_params=kwargs,
        )
        return await self._complete(req)

    def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[LLMStreamChunk]:
        req = ChatRequest(
            messages=tuple(messages),
            model=model or self.default_model,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_params=kwargs,
        )
        return self._stream(req)

    @abstractmethod
    async def _complete(self, req: ChatRequest) -> LLMResponse:
        raise NotImplementedError

    @abstractmethod
    def _stream(self, req: ChatRequest) -> AsyncIterator[LLMStreamChunk]:
        raise NotImplementedError
