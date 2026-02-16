"""Chat API: producer (persist + enqueue) + subscriber (SSE stream)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import cast
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse

from src.api.deps import CurrentUserDep, LLMRouterDep, RedisDep
from src.api.exceptions import _sse_event
from src.db import DbSessionDep
from src.models.message import MessageRole
from src.redis_client import enqueue_chat_request, events_stream_key
from src.repository import (
    ConversationRepository,
    LLMRequestRepository,
    MessageRepository,
)
from src.schemas import chat as schemas

router = APIRouter(prefix="/v1/chat", tags=["chat"])


@router.post("", response_model=schemas.ChatEnqueueResponse)
async def chat_enqueue(
    req: schemas.ChatEnqueueRequest,
    session: DbSessionDep,
    redis: RedisDep,
    llm_router: LLMRouterDep,
    current_user: CurrentUserDep,
) -> schemas.ChatEnqueueResponse:
    """
    Producer: persist user message + enqueue LLM request in one call.
    Returns request_id, user_message_id, assistant_message_id.
    Worker processes and emits events to Redis stream.
    """
    conversation_repo = ConversationRepository(session)
    message_repo = MessageRepository(session)
    llm_request_repo = LLMRequestRepository(session)

    conversation = await conversation_repo.get_by_id(req.conversation_id)
    if not conversation or conversation.user_id != current_user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found")

    # Idempotency: return existing if already processed
    existing = await llm_request_repo.get_by_client_request_id(
        req.conversation_id, req.client_request_id
    )
    if existing and existing.assistant_message_id and existing.user_message_id:
        assistant_msg = await message_repo.get_by_id(cast(UUID, existing.assistant_message_id))
        user_msg = await message_repo.get_by_id(cast(UUID, existing.user_message_id))
        return schemas.ChatEnqueueResponse(
            request_id=existing.id,
            user_message_id=cast(UUID, existing.user_message_id),
            user_seq=user_msg.seq if user_msg else 0,
            assistant_message_id=existing.assistant_message_id,
            assistant_seq=assistant_msg.seq if assistant_msg else 0,
            status=existing.status or "queued",
        )

    # Create or find user message (idempotent by client_msg_id)
    user_message = await message_repo.get_by_client_msg_id(req.conversation_id, req.client_msg_id)
    if not user_message:
        user_message = await message_repo.create(
            conversation_id=req.conversation_id,
            role=MessageRole.user,
            content=req.content,
            user_id=conversation.user_id,
            metadata=req.metadata,
            client_msg_id=req.client_msg_id,
        )
        await conversation_repo.update_on_message(
            req.conversation_id, user_message.id, user_message.seq
        )

    llm = llm_router.get(req.model)
    llm_request, assistant_placeholder = await llm_request_repo.create_with_placeholder(
        conversation_id=req.conversation_id,
        user_id=conversation.user_id,
        provider=llm.provider,
        model=req.model,
        user_message_id=user_message.id,
        snapshot_seq=user_message.seq,
        client_request_id=req.client_request_id,
        request_params=req.params,
        initial_status="queued",
    )
    await session.commit()

    await enqueue_chat_request(redis, str(llm_request.id))

    return schemas.ChatEnqueueResponse(
        request_id=llm_request.id,
        user_message_id=user_message.id,
        user_seq=user_message.seq,
        assistant_message_id=assistant_placeholder.id,
        assistant_seq=assistant_placeholder.seq,
        status="queued",
    )


@router.get("/stream")
async def chat_stream_subscribe(
    request: Request,
    session: DbSessionDep,
    redis: RedisDep,
    current_user: CurrentUserDep,
    request_id: UUID = Query(..., alias="request_id"),
    after_event_id: str = Query("0-0", alias="after_event_id"),
) -> StreamingResponse:
    """Subscriber: SSE from Redis chat:events:{request_id}. Reconnect via after_event_id or Last-Event-ID."""
    llm_request_repo = LLMRequestRepository(session)
    conversation_repo = ConversationRepository(session)
    llm_request = await llm_request_repo.get_by_id(request_id)
    if not llm_request:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Request not found")
    conversation = await conversation_repo.get_by_id(llm_request.conversation_id)
    if not conversation or conversation.user_id != current_user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")
    last_id = request.headers.get("Last-Event-ID") or after_event_id
    stream_key = events_stream_key(str(request_id))

    async def event_stream() -> AsyncGenerator[str, None]:
        nonlocal last_id
        try:
            while True:
                result = await redis.xread({stream_key: last_id}, block=15000, count=10)
                if not result:
                    yield ": keepalive\n\n"
                    continue

                for _, events in result:
                    for eid, raw_data in events:
                        last_id = eid
                        payload_str = (
                            raw_data.get("payload") if isinstance(raw_data, dict) else None
                        )
                        if not payload_str:
                            continue
                        try:
                            data = json.loads(payload_str)
                        except json.JSONDecodeError:
                            continue
                        event_type = data.get("type", "message")
                        sse_data = {k: v for k, v in data.items() if k != "type"}
                        yield _sse_event(event_type, sse_data)
                        if event_type == "usage" and sse_data.get("persisted"):
                            return
                        if event_type == "error":
                            return
        except asyncio.CancelledError:
            raise
        except Exception:
            yield _sse_event("error", {"error": "Stream read failed"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
