"""Redis client for chat queue and events (Redis Streams)."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from redis.asyncio import Redis

from src.utils.config import get_redis_url

logger = logging.getLogger(__name__)

# Stream keys (use chat:queue:test for integration tests to avoid clashing with real worker)
CHAT_QUEUE_STREAM = os.getenv("CHAT_QUEUE_STREAM", "chat:queue")
CHAT_EVENTS_STREAM_PREFIX = "chat:events:"


def events_stream_key(request_id: str) -> str:
    """Return Redis stream key for request events."""
    return f"{CHAT_EVENTS_STREAM_PREFIX}{request_id}"


async def create_redis_client() -> Redis:
    """Create async Redis client."""
    url = get_redis_url()
    return Redis.from_url(url, decode_responses=True)


async def close_redis_client(client: Redis) -> None:
    """Close Redis client."""
    await client.aclose()


async def enqueue_chat_request(redis: Redis, request_id: str) -> None:
    """Enqueue a chat request to the worker queue."""
    payload = json.dumps({"request_id": request_id})
    await redis.xadd(CHAT_QUEUE_STREAM, {"payload": payload}, "*")
    logger.info("chat_request_enqueued", extra={"request_id": request_id})


async def add_event(redis: Redis, request_id: str, event_type: str, data: dict[str, Any]) -> str:
    """Add an event to the request's events stream. Returns event id."""
    stream_key = events_stream_key(request_id)
    payload = json.dumps({"type": event_type, **data})
    event_id = await redis.xadd(stream_key, {"payload": payload}, "*")
    return event_id
