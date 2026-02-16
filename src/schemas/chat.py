from __future__ import annotations

from datetime import datetime
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


class LLMStreamChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    is_final: bool = False
    stats: LLMResponseStats | None = None
    raw: dict[str, Any] | None = None
    assistant_message_id: UUID | None = None
    assistant_seq: int | None = None
    persisted: bool = False


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

    title: str | None = None
    settings: dict[str, Any] = Field(default_factory=dict)


class CreateConversationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    conversation_id: UUID


class UpdateConversationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str | None = None


class ConversationListItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    title: str | None
    created_at: datetime
    last_message_at: datetime | None


class ListConversationsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversations: list[ConversationListItem]
    total: int


class StreamDoneEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assistant_message_id: UUID
    assistant_seq: int
    persisted: bool = True


# --- Queued chat (producer: persist message + enqueue) ---


class ChatEnqueueRequest(BaseModel):
    """Request body for POST /v1/chat. Persists user message and enqueues LLM request in one call."""

    model_config = ConfigDict(extra="forbid")

    conversation_id: UUID
    content: str
    client_msg_id: str
    client_request_id: str
    model: str
    params: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatEnqueueResponse(BaseModel):
    """Response from POST /v1/chat."""

    model_config = ConfigDict(extra="forbid")

    request_id: UUID
    user_message_id: UUID
    user_seq: int
    assistant_message_id: UUID
    assistant_seq: int
    status: str = "queued"
