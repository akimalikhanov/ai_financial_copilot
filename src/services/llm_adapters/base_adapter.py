from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class Role(str, Enum):
    system = "system"
    developer = "developer"
    user = "user"
    assistant = "assistant"
    tool = "tool"


@dataclass(frozen=True, slots=True)
class ToolCallRef:
    id: str
    name: str
    arguments: str  # raw JSON


@dataclass(frozen=True, slots=True)
class OpaqueItem:
    """A provider-native output item replayed verbatim on later calls (an encrypted
    reasoning item, say). Only the adapter that produced it reads it; tagged so an adapter
    can skip items from another provider or model."""

    provider: str
    model: str
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ResponseMeta:
    """Per-response facts for debugging, not usage."""

    response_id: str | None = None
    served_model: str | None = None
    reasoning_context: str | None = None  # effective mode the provider reports
    service_tier: str | None = None
    status: str | None = None
    incomplete_reason: str | None = None


@dataclass(frozen=True, slots=True)
class AssistantTurnResult:
    text: str
    tool_calls: list[ToolCallRef]
    stats: LLMResponseStats | None = None
    # The provider's stop reason, normalised to OpenAI's names: "stop", "tool_calls",
    # "length" (hit the completion-token cap), "content_filter", "refusal". None when the
    # adapter does not report it.
    finish_reason: str | None = None
    opaque: tuple[OpaqueItem, ...] | None = None
    reasoning_summary: str | None = None
    meta: ResponseMeta | None = None


@dataclass(frozen=True, slots=True)
class ImagePart:
    """One image attached to a message. Raw bytes, not base64: Gemini wants bytes and
    OpenAI wants a data URI, so each adapter encodes for its own wire format."""

    data: bytes
    mime_type: str  # "image/png" for the picture crops written in the ingestion pipeline
    # Resolution knob. Maps to OpenAI's image_url.detail and Gemini's Part.media_resolution;
    # "auto" leaves it to the provider. The main cost lever on image-heavy workloads.
    detail: Literal["auto", "low", "high"] = "auto"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Role
    content: str | None = None  # None valid for assistant tool-call turns
    name: str | None = None  # tool name (or function name)
    tool_call_id: str | None = None  # ties tool result to assistant tool call
    tool_calls: tuple[ToolCallRef, ...] | None = None
    # Images ride alongside content rather than widening it to a parts union, so the six
    # existing `m.content or ""` consumers keep working unchanged. complete() only.
    images: tuple[ImagePart, ...] | None = None
    # Assistant tool-call turns only: the provider items that produced the calls, in
    # emission order. Read only by the adapter; never traced, never crosses a user turn.
    opaque: tuple[OpaqueItem, ...] | None = None


def reject_images(messages: Sequence[ChatMessage], surface: str) -> None:
    """Raise if any message carries images. Only complete() supports them; failing here
    beats discovering months later that a description was silently text-only."""
    if any(m.images for m in messages):
        raise ValueError(f"{surface} does not support image content; use complete()")


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
    # Subset of input_tokens written to the provider's prompt cache.
    cache_write_tokens: int | None = None
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
    finish_reason: str | None = None  # as on AssistantTurnResult
    reasoning_summary: str | None = None
    meta: ResponseMeta | None = None


@dataclass(frozen=True, slots=True)
class LLMStreamChunk:
    text: str
    raw: Any = None  # provider-native chunk object (optional)
    is_final: bool = False
    stats: LLMResponseStats | None = None
    # Final chunk only.
    finish_reason: str | None = None
    reasoning_summary: str | None = None
    meta: ResponseMeta | None = None


class LLMAdapter(ABC):
    """Provider-agnostic non-streaming interface."""

    # Subclasses should override this to identify the provider for error mapping
    provider_name: str = "unknown"
    # Wire API, recorded per call (llm_requests.llm_api, llm_api_calls_total).
    api: str = "unknown"

    def __init__(self, *, default_model: str):
        self.default_model = default_model

    @abstractmethod
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
    ) -> AsyncGenerator[LLMStreamChunk, None]:
        reject_images(messages, f"{self.__class__.__name__}.stream")
        req = ChatRequest(
            messages=tuple(messages),
            model=model or self.default_model,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_params=kwargs,
        )
        return self._stream(req)

    async def complete_with_tools(
        self,
        messages: Sequence[ChatMessage],
        tools: list[dict],
        **kwargs: Any,
    ) -> AssistantTurnResult:
        # No reject_images guard here: this stub raises for every call already. The guard
        # goes in the subclasses that actually implement tool calling.
        raise NotImplementedError(f"{self.__class__.__name__} does not support tool calling")

    @abstractmethod
    async def _complete(self, req: ChatRequest) -> LLMResponse:
        raise NotImplementedError

    @abstractmethod
    def _stream(self, req: ChatRequest) -> AsyncGenerator[LLMStreamChunk, None]:
        raise NotImplementedError
