"""
Chat worker: Celery task that processes chat requests, calls LLM, emits events to chat:events:{request_id}.

Run as: .venv/bin/python -m src.workers.chat_worker
"""

from __future__ import annotations

import asyncio
import logging
import sys
from decimal import Decimal
from uuid import UUID

from celery.signals import worker_process_init, worker_process_shutdown
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from src.api.logging import configure_worker_logging
from src.celery_app import celery_app
from src.models.message import Message, MessageStatus
from src.redis_client import add_event, append_chat_tail, cas_populate_chat_tail, get_chat_tail
from src.repository import (
    ConversationRepository,
    LLMRequestRepository,
    MessageRepository,
)
from src.schemas import chat as schemas
from src.services.context import build_context
from src.services.llm_adapters.base_adapter import (
    ChatMessage as AdapterChatMessage,
)
from src.services.llm_adapters.base_adapter import Role as AdapterRole
from src.services.llm_router import get_router
from src.utils.config import get_chat_tail_max_messages, get_db_url, get_redis_app_url

logger = logging.getLogger(__name__)

_worker_loop: asyncio.AbstractEventLoop | None = None
_redis_app: Redis | None = None
_engine = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _initialize_worker_resources() -> None:
    global _worker_loop, _redis_app, _engine, _session_factory
    if _worker_loop is None or _worker_loop.is_closed():
        _worker_loop = asyncio.new_event_loop()
    if _redis_app is None:
        _redis_app = Redis.from_url(get_redis_app_url(), decode_responses=True)
    if _engine is None:
        _engine = create_async_engine(get_db_url(), poolclass=NullPool)
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            _engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )


def _get_worker_loop() -> asyncio.AbstractEventLoop:
    if _worker_loop is None or _worker_loop.is_closed():
        raise RuntimeError("Chat worker loop is not initialized")
    return _worker_loop


@worker_process_init.connect
def _on_worker_process_init(**_kwargs: object) -> None:
    configure_worker_logging()
    _initialize_worker_resources()


@worker_process_shutdown.connect
def _on_worker_process_shutdown(**_kwargs: object) -> None:
    global _worker_loop, _redis_app, _engine, _session_factory
    if _worker_loop is None or _worker_loop.is_closed():
        return
    if _redis_app is not None:
        _worker_loop.run_until_complete(_redis_app.aclose())
    if _engine is not None:
        _worker_loop.run_until_complete(_engine.dispose())
    _redis_app = None
    _engine = None
    _session_factory = None
    _worker_loop.close()
    _worker_loop = None


def _get_session_factory() -> async_sessionmaker[AsyncSession]:
    if _session_factory is None:
        raise RuntimeError("Chat worker DB session factory is not initialized")
    return _session_factory


def _get_redis_app() -> Redis:
    if _redis_app is None:
        raise RuntimeError("Chat worker Redis client is not initialized")
    return _redis_app


def _to_adapter_messages(messages: list[schemas.ChatMessage]) -> list[AdapterChatMessage]:
    return [
        AdapterChatMessage(
            role=AdapterRole(m.role),
            content=m.content,
            name=m.name,
            tool_call_id=m.tool_call_id,
        )
        for m in messages
    ]


def _error_event(exc: Exception, user_message: str | None = None) -> dict:
    """Structured error event for frontend display."""
    return {
        "error_type": type(exc).__name__,
        "message": str(exc),
        "user_message": user_message or str(exc),
    }


async def _process_chat_async(request_id: str) -> None:
    """Process a single chat request: load from DB, call LLM, stream to Redis."""
    redis_app = _get_redis_app()
    session_factory = _get_session_factory()

    async with session_factory() as session:
        llm_request_repo = LLMRequestRepository(session)
        message_repo = MessageRepository(session)
        conversation_repo = ConversationRepository(session)

        llm_request = await llm_request_repo.get_by_id(UUID(request_id))
        if not llm_request:
            logger.error("llm_request_not_found", extra={"request_id": request_id})
            await add_event(
                redis_app, request_id, "error", _error_event(LookupError("Request not found"))
            )
            return

        conversation_id = llm_request.conversation_id
        assistant_message_id = llm_request.assistant_message_id
        if not assistant_message_id:
            logger.error("no_assistant_placeholder", extra={"request_id": request_id})
            await add_event(
                redis_app,
                request_id,
                "error",
                _error_event(ValueError("No assistant placeholder")),
            )
            return

        assistant_msg = await message_repo.get_by_id(assistant_message_id)
        assistant_seq = assistant_msg.seq if assistant_msg else 0

        await llm_request_repo.update_status(UUID(request_id), "streaming")

        user_seq = llm_request.snapshot_seq or 0

        context_messages: list[schemas.ChatMessage] | None = None
        cached = await get_chat_tail(redis_app, str(conversation_id))
        if cached is not None:
            cached_msgs, cached_seq = cached
            if cached_seq >= user_seq:
                try:
                    context_messages = [schemas.ChatMessage.model_validate(m) for m in cached_msgs]
                except Exception:
                    logger.warning(
                        "invalid_chat_tail_cache",
                        extra={"conversation_id": str(conversation_id)},
                    )

        if context_messages is None:
            context_messages, latest_seq = await build_context(
                message_repo,
                conversation_id,
                before_seq=assistant_seq,
                max_messages=get_chat_tail_max_messages(),
            )
            await cas_populate_chat_tail(
                redis_app,
                str(conversation_id),
                [m.model_dump(mode="json") for m in context_messages],
                latest_seq,
            )

        adapter_messages = _to_adapter_messages(context_messages)

        params = dict(llm_request.request_params or {})
        temperature = params.get("temperature")
        max_tokens = params.get("max_tokens")

        try:
            llm = get_router().get(llm_request.model)
        except Exception as e:
            logger.exception("llm_router_error", extra={"request_id": request_id})
            await llm_request_repo.update_status(UUID(request_id), "failed")
            await add_event(redis_app, request_id, "error", _error_event(e))
            await session.commit()
            return

        accumulated_content = ""
        stream = llm.stream(
            messages=adapter_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **{k: v for k, v in params.items() if k not in ("temperature", "max_tokens")},
        )

        try:
            async for chunk in stream:
                accumulated_content += chunk.text
                if not chunk.is_final:
                    await add_event(redis_app, request_id, "delta", {"text": chunk.text})
                    continue

                await message_repo.update_on_final(
                    message_id=assistant_message_id,
                    content=accumulated_content,
                    request_id=UUID(request_id),
                )
                if chunk.stats:
                    await llm_request_repo.update_on_final(
                        request_id=UUID(request_id),
                        prompt_tokens=chunk.stats.input_tokens,
                        completion_tokens=chunk.stats.output_tokens,
                        reasoning_tokens=chunk.stats.reasoning_tokens,
                        total_tokens=chunk.stats.total_tokens,
                        cost_usd=(
                            Decimal(str(chunk.stats.cost_usd))
                            if chunk.stats.cost_usd is not None
                            else None
                        ),
                        latency_ms=(
                            int(chunk.stats.latency_ms)
                            if chunk.stats.latency_ms is not None
                            else None
                        ),
                        ttft_ms=(
                            int(chunk.stats.ttft_ms) if chunk.stats.ttft_ms is not None else None
                        ),
                        tps=(int(chunk.stats.tps) if chunk.stats.tps is not None else None),
                    )
                await llm_request_repo.update_status(UUID(request_id), "completed")
                await conversation_repo.update_on_message(
                    conversation_id=conversation_id,
                    message_id=assistant_message_id,
                    new_seq=assistant_seq,
                )

                usage_data = {
                    "persisted": True,
                    "assistant_message_id": str(assistant_message_id),
                    "assistant_seq": assistant_seq,
                }
                if chunk.stats:
                    usage_data["stats"] = {
                        "input_tokens": chunk.stats.input_tokens,
                        "output_tokens": chunk.stats.output_tokens,
                        "reasoning_tokens": chunk.stats.reasoning_tokens,
                        "total_tokens": chunk.stats.total_tokens,
                        "latency_ms": chunk.stats.latency_ms,
                        "ttft_ms": chunk.stats.ttft_ms,
                        "tps": chunk.stats.tps,
                        "cost_usd": chunk.stats.cost_usd,
                    }
                await add_event(redis_app, request_id, "usage", usage_data)
                await session.commit()

                try:
                    await append_chat_tail(
                        redis_app,
                        str(conversation_id),
                        schemas.ChatMessage(
                            role=schemas.Role.assistant,
                            content=accumulated_content,
                        ).model_dump(mode="json"),
                        assistant_seq,
                    )
                except Exception:
                    logger.warning(
                        "chat_tail_append_failed",
                        extra={"request_id": request_id},
                    )

        except Exception as e:
            logger.exception("llm_stream_error", extra={"request_id": request_id})
            await llm_request_repo.update_status(UUID(request_id), "failed")
            await llm_request_repo.update_on_final(
                request_id=UUID(request_id),
                error_code=type(e).__name__,
                error_message=str(e),
            )
            from sqlalchemy import select

            result = await session.execute(
                select(Message).where(Message.id == assistant_message_id)
            )
            msg = result.scalar_one_or_none()
            if msg:
                msg.status = MessageStatus.error
            await add_event(redis_app, request_id, "error", _error_event(e, str(e)))
            await session.commit()


@celery_app.task(bind=True, name="process_chat", acks_late=True, reject_on_worker_lost=True)
def process_chat(_self, request_id: str) -> None:
    """Celery task: process chat request."""
    _initialize_worker_resources()
    loop = _get_worker_loop()
    loop.run_until_complete(_process_chat_async(request_id))


def main() -> None:
    configure_worker_logging()
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    celery_app.worker_main(argv=["worker", "-Q", "chat", "--pool=prefork"])


if __name__ == "__main__":
    main()
