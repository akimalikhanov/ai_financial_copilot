from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class Role(str, Enum):
    system = "system"
    developer = "developer"
    user = "user"
    assistant = "assistant"
    tool = "tool"


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Role
    content: str
    name: str | None = None
    tool_call_id: str | None = None


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessage]
    model: str
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    extra_params: dict[str, Any] = Field(default_factory=dict)
    conversation_id: UUID | None = None


class LLMResponseStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: float | None = None
    ttft_ms: float | None = None
    tps: float | None = None
    cost_usd: float | None = None


class LLMResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    stats: LLMResponseStats | None = None
    raw: dict[str, Any] | None = None


class LLMStreamChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    is_final: bool = False
    stats: LLMResponseStats | None = None
    raw: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error_type: str
    message: str
    internal_message: str | None = None
    user_message: str | None = None
    provider: str | None = None
    model: str | None = None
    is_retryable: bool | None = None
    status_code: int | None = None
    error_code: str | None = None
    original_error_message: str | None = None


class CreateConversationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: UUID | None = None
    title: str | None = None
    settings: dict[str, Any] = Field(default_factory=dict)


class CreateConversationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    conversation_id: UUID


class UpdateConversationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str | None = None


class CreateMessageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    conversation_id: UUID
    role: Role
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class CreateMessageBody(BaseModel):
    """Request body for creating a message (conversation_id comes from path)."""

    model_config = ConfigDict(extra="forbid")
    role: Role
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class CreateMessageResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message_id: UUID
    seq: int
