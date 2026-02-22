"""
Chat worker: consumes from chat:queue, calls LLM, emits events to chat:events:{request_id}.

Run as: .venv/bin/python -m src.workers.chat_worker
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
from decimal import Decimal
from uuid import UUID

from redis.asyncio import Redis

from src.api.logging import configure_worker_logging
from src.db import get_session_factory, init_db, shutdown_db
from src.models.message import Message, MessageStatus
from src.redis_client import CHAT_QUEUE_STREAM, events_stream_key
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
from src.utils.config import get_redis_app_url, get_redis_broker_url

logger = logging.getLogger(__name__)


CONSUMER_GROUP = "chat-workers"
WORKER_ID = os.getenv("CHAT_WORKER_ID", "worker-1")
BLOCK_MS = int(os.getenv("CHAT_WORKER_BLOCK_MS", "15000"))


async def run_consume_loop(
    redis_broker: Redis,
    redis_app: Redis,
    llm_router,
    shutdown: asyncio.Event,
    *,
    block_ms: int = 500,
) -> None:
    """
    Consume jobs from chat queue until shutdown is set.
    Does NOT init/shutdown DB or close redis (for integration test use).
    """
    try:
        await redis_broker.xgroup_create(CHAT_QUEUE_STREAM, CONSUMER_GROUP, id="$", mkstream=True)
    except Exception as e:
        if "BUSYGROUP" not in str(e) and "already exists" not in str(e).lower():
            logger.warning("consumer_group_create", extra={"error": str(e)})

    while not shutdown.is_set():
        try:
            result = await redis_broker.xreadgroup(
                groupname=CONSUMER_GROUP,
                consumername=WORKER_ID,
                streams={CHAT_QUEUE_STREAM: ">"},
                block=block_ms,
                count=1,
            )
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.warning("xreadgroup_error", extra={"error": str(e)})
            await asyncio.sleep(0.1)
            continue

        if not result:
            continue

        for _stream_name, messages in result:
            for msg_id, raw in messages:
                payload_str = raw.get("payload") if isinstance(raw, dict) else ""
                try:
                    data = json.loads(payload_str or "")
                    rid = data.get("request_id")
                except (json.JSONDecodeError, KeyError):
                    await redis_broker.xack(CHAT_QUEUE_STREAM, CONSUMER_GROUP, msg_id)
                    continue

                if not rid:
                    await redis_broker.xack(CHAT_QUEUE_STREAM, CONSUMER_GROUP, msg_id)
                    continue

                try:
                    await process_request(redis_app, rid, llm_router)
                finally:
                    await redis_broker.xack(CHAT_QUEUE_STREAM, CONSUMER_GROUP, msg_id)


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


async def add_event(redis: Redis, request_id: str, event_type: str, data: dict) -> None:
    """Add event to request's event stream."""
    stream_key = events_stream_key(request_id)
    payload = json.dumps({"type": event_type, **data})
    await redis.xadd(stream_key, {"payload": payload}, "*")


async def process_request(
    redis_app: Redis,
    request_id: str,
    llm_router,
) -> None:
    """Process a single chat request: load from DB, call LLM, stream to Redis."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        llm_request_repo = LLMRequestRepository(session)
        message_repo = MessageRepository(session)
        conversation_repo = ConversationRepository(session)

        llm_request = await llm_request_repo.get_by_id(UUID(request_id))
        if not llm_request:
            logger.error("llm_request_not_found", extra={"request_id": request_id})
            await add_event(redis_app, request_id, "error", {"error": "Request not found"})
            return

        conversation_id = llm_request.conversation_id
        assistant_message_id = llm_request.assistant_message_id
        if not assistant_message_id:
            logger.error("no_assistant_placeholder", extra={"request_id": request_id})
            await add_event(redis_app, request_id, "error", {"error": "No assistant placeholder"})
            return

        assistant_msg = await message_repo.get_by_id(assistant_message_id)
        assistant_seq = assistant_msg.seq if assistant_msg else 0

        # Update status to streaming
        await llm_request_repo.update_status(UUID(request_id), "streaming")

        # Build context (same sliding window as producer)
        context_messages, _ = await build_context(
            message_repo, conversation_id, after_seq=None, max_messages=50
        )
        adapter_messages = _to_adapter_messages(context_messages)

        params = dict(llm_request.request_params or {})
        temperature = params.get("temperature")
        max_tokens = params.get("max_tokens")

        try:
            llm = llm_router.get(llm_request.model)
        except Exception as e:
            logger.exception("llm_router_error", extra={"request_id": request_id})
            await llm_request_repo.update_status(UUID(request_id), "failed")
            await add_event(redis_app, request_id, "error", {"error": str(e)})
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

                # Final chunk: update DB, emit usage event
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
            await add_event(redis_app, request_id, "error", {"error": str(e)})
            await session.commit()


async def run_worker() -> None:
    """Main worker loop."""
    configure_worker_logging()
    await init_db()
    redis_broker = Redis.from_url(get_redis_broker_url(), decode_responses=True)
    redis_app = Redis.from_url(get_redis_app_url(), decode_responses=True)
    llm_router = get_router()

    # Create consumer group (ignore error if already exists)
    # $ = only new messages from now; use "0" to process from start
    try:
        await redis_broker.xgroup_create(CHAT_QUEUE_STREAM, CONSUMER_GROUP, id="$", mkstream=True)
        logger.info("consumer_group_created", extra={"group": CONSUMER_GROUP})
    except Exception as e:
        if "BUSYGROUP" not in str(e) and "already exists" not in str(e).lower():
            logger.warning("consumer_group_create", extra={"error": str(e)})

    shutdown = asyncio.Event()

    def on_signal(*_args):
        shutdown.set()

    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, on_signal)
    except (ValueError, OSError):
        pass  # signal.signal not available in all contexts (e.g. Windows)

    logger.info("chat_worker_started", extra={"worker_id": WORKER_ID})

    try:
        while not shutdown.is_set():
            try:
                result = await redis_broker.xreadgroup(
                    groupname=CONSUMER_GROUP,
                    consumername=WORKER_ID,
                    streams={CHAT_QUEUE_STREAM: ">"},
                    block=BLOCK_MS,
                    count=1,
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("xreadgroup_error", extra={"error": str(e)})
                await asyncio.sleep(1)
                continue

            if not result:
                continue

            for _stream_name, messages in result:
                for msg_id, raw in messages:
                    payload_str = raw.get("payload") if isinstance(raw, dict) else ""
                    try:
                        data = json.loads(payload_str or "")
                        rid = data.get("request_id")
                    except (json.JSONDecodeError, KeyError):
                        logger.warning("invalid_queue_payload", extra={"msg_id": msg_id})
                        await redis_broker.xack(CHAT_QUEUE_STREAM, CONSUMER_GROUP, msg_id)
                        continue

                    if not rid:
                        await redis_broker.xack(CHAT_QUEUE_STREAM, CONSUMER_GROUP, msg_id)
                        continue

                    try:
                        await process_request(redis_app, rid, llm_router)
                    finally:
                        await redis_broker.xack(CHAT_QUEUE_STREAM, CONSUMER_GROUP, msg_id)

    finally:
        await redis_broker.aclose()
        await redis_app.aclose()
        await shutdown_db()
        logger.info("chat_worker_stopped")


def main() -> None:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
