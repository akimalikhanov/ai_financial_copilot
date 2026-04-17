from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from redis.asyncio import Redis
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.models.llm_request import LLMRequest
    from src.schemas.query_router import DocumentScopeResult, RouterOutput
    from src.schemas.query_transform import TransformedQuery
    from src.schemas.retrieval import ProcessedQuery, RAGContext
    from src.services.context.conversation_history import ConversationHistory
    from src.services.llm_adapters.base_adapter import ChatMessage as AdapterChatMessage


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


@dataclass
class ChatPipelineState:
    """Pipeline state passed between chat pipeline stages."""

    request_id: str
    redis_app: Redis
    session: AsyncSession
    llm_request: LLMRequest | None = None
    conversation_id: UUID | None = None
    assistant_message_id: UUID | None = None
    assistant_seq: int = 0
    history: ConversationHistory | None = None
    context_messages: list[ChatMessage] | None = None
    user_query_raw: str = ""
    processed_query: ProcessedQuery | None = None
    router_output: RouterOutput | None = None
    scope_result: DocumentScopeResult | None = None
    transformed_query: TransformedQuery | None = None
    rag_context: RAGContext | None = None
    rag_context_str: str = ""
    adapter_messages: list[AdapterChatMessage] | None = None
    accumulated_content: str = ""
    clean_content: str = ""
    params: dict = field(default_factory=dict)


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


# --- Stats (GET /v1/chat/stats) ---


class RequestStatsItem(BaseModel):
    """Single LLM request stats for stats API."""

    model_config = ConfigDict(extra="forbid")

    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None
    latency_ms: int | None = None
    ttft_ms: int | None = None
    tps: int | None = None
    model: str
    created_at: datetime

    # Full pipeline aggregates (chat LLM + router sub-requests combined).
    # Use these for total cost and token count; use the fields above for
    # per-model breakdowns (token distribution bar, latency, TPS).
    pipeline_cost_usd: float | None = None
    pipeline_total_tokens: int | None = None


class ChatStatsResponse(BaseModel):
    """Response from GET /v1/chat/stats."""

    model_config = ConfigDict(extra="forbid")

    requests: list[RequestStatsItem]
