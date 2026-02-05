from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import aclosing
from decimal import Decimal
from typing import TYPE_CHECKING
from uuid import UUID

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from src.api.deps import LLMRouterDep
from src.api.exceptions import _error_to_schema, _sse_event
from src.api.logging import set_request_error, set_request_meta
from src.db import DbSessionDep
from src.models.message import MessageRole
from src.repository import (
    ConversationRepository,
    LLMRequestRepository,
    MessageRepository,
)

if TYPE_CHECKING:
    from src.models.message import Message
from src.schemas import chat as schemas
from src.services.llm_adapters.base_adapter import (
    ChatMessage as AdapterChatMessage,
)
from src.services.llm_adapters.base_adapter import (
    LLMResponse as AdapterResponse,
)
from src.services.llm_adapters.base_adapter import (
    LLMResponseStats as AdapterResponseStats,
)
from src.services.llm_adapters.base_adapter import (
    LLMStreamChunk as AdapterStreamChunk,
)
from src.services.llm_adapters.base_adapter import (
    Role as AdapterRole,
)
from src.services.llm_runtime.exceptions import LLMError

router = APIRouter(prefix="/v1/chat", tags=["chat"])


def _to_adapter_messages(messages: list[schemas.ChatMessage]) -> list[AdapterChatMessage]:
    return [
        AdapterChatMessage(
            role=AdapterRole(message.role),
            content=message.content,
            name=message.name,
            tool_call_id=message.tool_call_id,
        )
        for message in messages
    ]


def _stats_to_schema(stats: AdapterResponseStats | None) -> schemas.LLMResponseStats | None:
    if stats is None:
        return None
    return schemas.LLMResponseStats(
        input_tokens=stats.input_tokens,
        cached_input_tokens=stats.cached_input_tokens,
        output_tokens=stats.output_tokens,
        reasoning_tokens=stats.reasoning_tokens,
        total_tokens=stats.total_tokens,
        latency_ms=stats.latency_ms,
        ttft_ms=stats.ttft_ms,
        tps=stats.tps,
        cost_usd=stats.cost_usd,
    )


def _response_to_schema(response: AdapterResponse) -> schemas.LLMResponse:
    return schemas.LLMResponse(
        text=response.text,
        stats=_stats_to_schema(response.stats),
        raw=None,
    )


def _stream_chunk_to_schema(chunk: AdapterStreamChunk) -> schemas.LLMStreamChunk:
    return schemas.LLMStreamChunk(
        text=chunk.text,
        is_final=chunk.is_final,
        stats=_stats_to_schema(chunk.stats),
        raw=None,
    )


def _message_role_to_schema_role(message_role: MessageRole) -> schemas.Role:
    """Convert MessageRole enum to schema Role enum."""
    role_map = {
        MessageRole.system: schemas.Role.system,
        MessageRole.user: schemas.Role.user,
        MessageRole.assistant: schemas.Role.assistant,
        MessageRole.tool: schemas.Role.tool,
    }
    return role_map[message_role]


def _db_message_to_chat_message(message: Message) -> schemas.ChatMessage:
    """Convert database Message model to ChatMessage schema."""
    return schemas.ChatMessage(
        role=_message_role_to_schema_role(message.role),
        content=message.content,
    )


@router.post("", response_model=schemas.LLMResponse, response_model_exclude_none=True)
async def chat(
    req: schemas.ChatRequest,
    llm_router: LLMRouterDep,
) -> schemas.LLMResponse:
    messages = _to_adapter_messages(req.messages)
    llm = llm_router.get(req.model)
    # Enrich request log with LLM metadata
    set_request_meta(provider=llm.provider, model_id=llm.model_id)
    response = await llm.complete(
        messages=messages,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        **req.extra_params,
    )
    return _response_to_schema(response)


@router.post("/stream")
async def chat_stream(
    req: schemas.ChatRequest,
    llm_router: LLMRouterDep,
    session: DbSessionDep,
) -> StreamingResponse:
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }

    llm = llm_router.get(req.model)
    # Enrich request log with LLM metadata
    set_request_meta(provider=llm.provider, model_id=llm.model_id)

    # Handle conversation_id if provided
    llm_request_id: UUID | None = None
    conversation_id: UUID | None = req.conversation_id
    assistant_message_id: UUID | None = None
    accumulated_content = ""
    conversation_user_id: UUID | None = None

    if conversation_id:
        # Create repositories
        conversation_repo = ConversationRepository(session)
        message_repo = MessageRepository(session)
        llm_request_repo = LLMRequestRepository(session)

        # Verify conversation exists and get user_id
        conversation = await conversation_repo.get_by_id(conversation_id)
        if not conversation:
            # Return error - conversation not found
            error_payload = schemas.ErrorResponse(
                error_type="NotFound",
                message=f"Conversation {conversation_id} not found",
            ).model_dump(exclude_none=True)
            return StreamingResponse(
                _sse_event("error", error_payload),
                media_type="text/event-stream",
                headers=headers,
            )

        # Store user_id for use in closure
        conversation_user_id = conversation.user_id

        # Create LLM request record
        request_params = {
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
            **req.extra_params,
        }
        llm_request = await llm_request_repo.create(
            conversation_id=conversation_id,
            user_id=conversation_user_id,  # Can be None for skip-auth approach
            provider=llm.provider,
            model=llm.model_id,
            request_params=request_params,
        )
        llm_request_id = llm_request.id

        # Fetch all messages for conversation
        db_messages = await message_repo.get_by_conversation_id(conversation_id)

        # Find the last user message and link it to LLM request
        last_user_message = None
        for msg in reversed(db_messages):
            if msg.role == MessageRole.user:
                last_user_message = msg
                break

        if last_user_message:
            last_user_message.request_id = llm_request_id
            await session.flush()

        # Convert DB messages to ChatMessage format
        messages = [_db_message_to_chat_message(msg) for msg in db_messages]
    else:
        # Backward compatibility: use messages from request
        messages = req.messages

    # Convert to adapter messages
    adapter_messages = _to_adapter_messages(messages)

    # Stream to LLM
    stream = llm.stream(
        messages=adapter_messages,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        **req.extra_params,
    )

    async def event_stream() -> AsyncGenerator[str, None]:
        nonlocal accumulated_content, assistant_message_id

        # Use aclosing to ensure stream cleanup on client disconnect or errors
        async with aclosing(stream) as safe_stream:
            try:
                async for chunk in safe_stream:
                    accumulated_content += chunk.text
                    event = "usage" if chunk.is_final else "delta"
                    payload = _stream_chunk_to_schema(chunk).model_dump(exclude_none=True)
                    yield _sse_event(event, payload)

                    # On final chunk, save to database if conversation_id provided
                    if chunk.is_final and conversation_id and llm_request_id:
                        message_repo = MessageRepository(session)
                        conversation_repo = ConversationRepository(session)
                        llm_request_repo = LLMRequestRepository(session)

                        # Get conversation to update message_count
                        conversation = await conversation_repo.get_by_id(conversation_id)
                        if not conversation:
                            continue  # Skip DB updates if conversation not found

                        # Create assistant message
                        assistant_message = await message_repo.create(
                            conversation_id=conversation_id,
                            role=MessageRole.assistant,
                            content=accumulated_content,
                            user_id=conversation_user_id,
                            request_id=llm_request_id,
                        )
                        assistant_message_id = assistant_message.id

                        # Update LLM request with stats
                        if chunk.stats:
                            await llm_request_repo.update_on_final(
                                request_id=llm_request_id,
                                prompt_tokens=chunk.stats.input_tokens,
                                completion_tokens=chunk.stats.output_tokens,
                                reasoning_tokens=chunk.stats.reasoning_tokens,
                                total_tokens=chunk.stats.total_tokens,
                                cost_usd=Decimal(str(chunk.stats.cost_usd))
                                if chunk.stats.cost_usd is not None
                                else None,
                                latency_ms=int(chunk.stats.latency_ms)
                                if chunk.stats.latency_ms is not None
                                else None,
                                ttft_ms=int(chunk.stats.ttft_ms)
                                if chunk.stats.ttft_ms is not None
                                else None,
                                tps=int(chunk.stats.tps) if chunk.stats.tps is not None else None,
                            )

                        # Update conversation stats
                        await conversation_repo.update_on_message(
                            conversation_id=conversation_id,
                            message_id=assistant_message_id,
                            message_count=conversation.message_count + 1,
                        )

            except LLMError as exc:
                # Errors during streaming are handled here (can't be caught by exception handler)
                set_request_error(error_type=type(exc).__name__, error_msg=str(exc))

                # Update LLM request with error if conversation_id provided
                if conversation_id and llm_request_id:
                    llm_request_repo = LLMRequestRepository(session)
                    await llm_request_repo.update_on_final(
                        request_id=llm_request_id,
                        error_code=type(exc).__name__,
                        error_message=str(exc),
                    )

                error_payload = _error_to_schema(exc).model_dump(exclude_none=True)
                yield _sse_event("error", error_payload)
            except Exception as exc:
                # Catch-all for unexpected errors during streaming
                set_request_error(error_type="InternalServerError", error_msg=str(exc))

                # Update LLM request with error if conversation_id provided
                if conversation_id and llm_request_id:
                    llm_request_repo = LLMRequestRepository(session)
                    await llm_request_repo.update_on_final(
                        request_id=llm_request_id,
                        error_code="InternalServerError",
                        error_message=str(exc),
                    )

                error_payload = schemas.ErrorResponse(
                    error_type="InternalServerError",
                    message="Internal server error",
                ).model_dump(exclude_none=True)
                yield _sse_event("error", error_payload)

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)
