"""Redis client for chat queue and events (Redis Streams)."""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from redis.asyncio import Redis

from src.utils.config import (
    get_chat_queue_stream,
    get_rate_limit_max_requests,
    get_rate_limit_window_ms,
    get_redis_app_url,
    get_redis_broker_url,
)

CHAT_QUEUE_STREAM = get_chat_queue_stream()
CHAT_EVENTS_STREAM_PREFIX = "chat:events:"


def events_stream_key(request_id: str) -> str:
    """Return Redis stream key for request events."""
    return f"{CHAT_EVENTS_STREAM_PREFIX}{request_id}"


async def create_redis_app_client() -> Redis:
    """Create async Redis client for app (rate limit, cache, SSE stream)."""
    return Redis.from_url(get_redis_app_url(), decode_responses=True)


async def create_redis_broker_client() -> Redis:
    """Create async Redis client for broker (chat:queue, PDF tasks)."""
    return Redis.from_url(get_redis_broker_url(), decode_responses=True)


async def close_redis_client(client: Redis) -> None:
    """Close Redis client."""
    await client.aclose()


async def enqueue_chat_request(redis: Redis, request_id: str) -> None:
    """Enqueue a chat request to the worker queue."""
    payload = json.dumps({"request_id": request_id})
    await redis.xadd(CHAT_QUEUE_STREAM, {"payload": payload}, "*")


async def add_event(redis: Redis, request_id: str, event_type: str, data: dict[str, Any]) -> str:
    """Add an event to the request's events stream. Returns event id."""
    stream_key = events_stream_key(request_id)
    payload = json.dumps({"type": event_type, **data})
    event_id = await redis.xadd(stream_key, {"payload": payload}, "*")
    return event_id


# Sliding-window rate limit for chat (LLM cost protection).
# Uses Redis server time so the limiter is consistent across all API nodes (no clock drift).
_SLIDING_WINDOW_LUA = """
local key = KEYS[1]
local t = redis.call('TIME')
local now_ms = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
local window_ms = tonumber(ARGV[1])
local limit = tonumber(ARGV[2])
local member = ARGV[3]
redis.call('ZREMRANGEBYSCORE', key, '-inf', now_ms - window_ms)
local count = redis.call('ZCARD', key)
if count >= limit then
  return {0, count}
end
redis.call('ZADD', key, now_ms, member)
redis.call('PEXPIRE', key, window_ms + 60000)
return {1, count + 1}
"""


# Script is cached per Redis client (keyed by id) to satisfy type checkers and avoid dynamic attrs
_rate_limit_script_cache: dict[int, Any] = {}


def _get_rate_limit_script(redis: Redis) -> Any:
    """Get or create the rate limit script (cached per client)."""
    k = id(redis)
    if k not in _rate_limit_script_cache:
        _rate_limit_script_cache[k] = redis.register_script(_SLIDING_WINDOW_LUA)
    return _rate_limit_script_cache[k]


async def check_chat_rate_limit(redis: Redis, user_id: str) -> tuple[bool, int]:
    """
    Sliding-window rate limit for chat enqueue. Returns (allowed, current_count).
    If not allowed, caller should return 429 with Retry-After header.
    """
    window_ms = get_rate_limit_window_ms()
    limit = get_rate_limit_max_requests()
    key = f"ratelimit:chat:{user_id}"
    member = str(uuid4())
    script = _get_rate_limit_script(redis)
    result = await script(keys=[key], args=[window_ms, limit, member])
    allowed = int(result[0]) == 1
    count = int(result[1])
    return allowed, count
