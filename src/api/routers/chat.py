"""Chat API: producer (persist + enqueue) + subscriber (SSE stream)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from time import perf_counter
from typing import cast
from uuid import UUID

from celery import Task
from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select

from src.api.deps import CurrentUserDep, LLMRouterDep, RedisDep, chat_rate_limit
from src.api.exceptions import _sse_event
from src.db import DbSessionDep, get_session_factory
from src.models.llm_request import LLMRequest
from src.models.message import MessageRole
from src.observability.metrics import (
    SSE_STREAM_DURATION,
    sse_stream_closed,
    sse_stream_opened,
)
from src.redis_client import events_stream_key
from src.repository import (
    ConversationRepository,
    LLMRequestRepository,
    MessageRepository,
)
from src.schemas import chat as schemas
from src.services.chat.tasks import process_chat
from src.services.context import ConversationHistory

router = APIRouter(prefix="/v1/chat", tags=["chat"])


@router.get("/stats", response_model=schemas.ChatStatsResponse)
async def chat_stats(
    session: DbSessionDep,
    current_user: CurrentUserDep,
    conversation_id: UUID = Query(..., alias="conversation_id"),
    limit: int = Query(50, ge=1, le=100),
) -> schemas.ChatStatsResponse:
    """Return recent LLM request stats for a conversation (conversation-scoped)."""
    conversation_repo = ConversationRepository(session)
    conversation = await conversation_repo.get_by_id(conversation_id)
    if not conversation or conversation.user_id != current_user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found")

    llm_request_repo = LLMRequestRepository(session)
    requests = await llm_request_repo.list_recent_by_conversation_id(conversation_id, limit=limit)

    # Aggregate sub-request (router, etc.) cost+tokens per parent in one query.
    parent_ids = [r.id for r in requests]
    sub_totals: dict[UUID, tuple[float, int]] = {}
    if parent_ids:
        rows = await session.execute(
            select(
                LLMRequest.parent_request_id,
                func.sum(LLMRequest.cost_usd).label("cost_sum"),
                func.sum(LLMRequest.total_tokens).label("token_sum"),
            )
            .where(LLMRequest.parent_request_id.in_(parent_ids))
            .group_by(LLMRequest.parent_request_id)
        )
        for row in rows:
            sub_totals[row.parent_request_id] = (
                float(row.cost_sum or 0),
                int(row.token_sum or 0),
            )

    items = []
    for r in requests:
        chat_cost = float(r.cost_usd) if r.cost_usd is not None else None
        chat_tokens = r.total_tokens
        sub_cost, sub_tokens = sub_totals.get(r.id, (0.0, 0))
        items.append(
            schemas.RequestStatsItem(
                input_tokens=r.prompt_tokens,
                output_tokens=r.completion_tokens,
                reasoning_tokens=r.reasoning_tokens,
                total_tokens=chat_tokens,
                cost_usd=chat_cost,
                latency_ms=r.latency_ms,
                ttft_ms=r.ttft_ms,
                tps=r.tps,
                model=r.model,
                created_at=r.created_at,
                pipeline_cost_usd=(chat_cost or 0.0) + sub_cost
                if chat_cost is not None or sub_cost
                else None,
                pipeline_total_tokens=(chat_tokens or 0) + sub_tokens
                if chat_tokens is not None or sub_tokens
                else None,
            )
        )
    return schemas.ChatStatsResponse(requests=items)


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
    await chat_rate_limit(redis, current_user)
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

    history = ConversationHistory(redis, message_repo)
    await history.append_user(req.conversation_id, req.content, user_message.seq)

    cast(Task, process_chat).delay(str(llm_request.id))

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

    # Release the pgbouncer server connection before the stream starts. `session` is injected
    # via get_current_user whether or not this signature asks for it, and FastAPI holds yield
    # dependencies until the streaming body finishes — so this commit is the only thing that
    # ends the transaction. Everything below needs only plain UUIDs.
    await session.commit()

    last_id = request.headers.get("Last-Event-ID") or after_event_id
    stream_key = events_stream_key(str(request_id))
    session_factory = get_session_factory()

    async def _failure_message() -> str | None:
        """Short-lived session for the liveness re-check — opened only when needed."""
        async with session_factory() as s:
            req = await LLMRequestRepository(s).get_by_id(request_id)
            if req and req.status == "failed":
                return req.error_message or "Processing failed"
            return None

    async def event_stream() -> AsyncGenerator[str, None]:
        nonlocal last_id
        empty_polls = 0
        sse_stream_opened("chat")
        _started = perf_counter()
        outcome = "client_closed"
        try:
            yield ": ok\n\n"
            while True:
                result = await redis.xread({stream_key: last_id}, block=15000, count=10)
                if not result:
                    empty_polls += 1
                    if empty_polls >= 3:
                        failure_message = await _failure_message()
                        if failure_message is not None:
                            outcome = "error"
                            yield _sse_event(
                                "error",
                                {
                                    "error_type": "WorkerError",
                                    "message": failure_message,
                                },
                            )
                            return
                    yield ": keepalive\n\n"
                    continue

                empty_polls = 0
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
                        yield _sse_event(event_type, sse_data, event_id=eid)
                        if event_type == "usage" and sse_data.get("persisted"):
                            outcome = "complete"
                            return
                        if event_type == "error":
                            outcome = "error"
                            return
        except asyncio.CancelledError:
            raise
        except Exception:
            outcome = "error"
            yield _sse_event("error", {"error": "Stream read failed"})
        finally:
            sse_stream_closed("chat")
            SSE_STREAM_DURATION.labels("chat", outcome).observe(perf_counter() - _started)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
