"""Redis client for chat events (Redis Streams) and chat history cache."""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from redis.asyncio import Redis

from src.utils.config import (
    get_chat_tail_max_messages,
    get_chat_tail_ttl,
    get_rate_limit_max_requests,
    get_rate_limit_window_ms,
    get_redis_app_url,
)

CHAT_EVENTS_STREAM_PREFIX = "chat:events:"
CHAT_TAIL_KEY_PREFIX = "chat:tail:"
INGESTION_STREAM_PREFIX = "ingestion:events:"


def events_stream_key(request_id: str) -> str:
    """Return Redis stream key for request events."""
    return f"{CHAT_EVENTS_STREAM_PREFIX}{request_id}"


def ingestion_stream_key(document_id: str) -> str:
    """Return Redis stream key for ingestion events."""
    return f"{INGESTION_STREAM_PREFIX}{document_id}"


async def create_redis_app_client() -> Redis:
    """Create async Redis client for app (rate limit, cache, SSE stream)."""
    return Redis.from_url(get_redis_app_url(), decode_responses=True)


async def close_redis_client(client: Redis) -> None:
    """Close Redis client."""
    await client.aclose()


def _chat_tail_key(conversation_id: str) -> str:
    return f"{CHAT_TAIL_KEY_PREFIX}{conversation_id}"


_CAS_POPULATE_LUA = """
-- KEYS[1] = chat:tail:{conv_id}
-- ARGV[1] = full JSON payload string
-- ARGV[2] = new latest_seq
-- ARGV[3] = TTL seconds
-- Returns: 1 if written, 0 if skipped
local key = KEYS[1]
local new_seq = tonumber(ARGV[2])
local ttl = tonumber(ARGV[3])

local current = redis.call('GET', key)
if current ~= false then
  local ok, data = pcall(cjson.decode, current)
  if ok and type(data) == 'table' then
    local cur_seq = tonumber(data['latest_seq'])
    if cur_seq and cur_seq >= new_seq then
      redis.call('EXPIRE', key, ttl)
      return 0
    end
  end
end

redis.call('SET', key, ARGV[1], 'EX', ttl)
return 1
"""

_APPEND_TRIM_LUA = """
-- KEYS[1] = chat:tail:{conv_id}
-- ARGV[1] = JSON string of the single message
-- ARGV[2] = seq of this message
-- ARGV[3] = max messages to keep
-- ARGV[4] = TTL seconds
-- Returns: 1 if appended, 0 if skipped
local key = KEYS[1]
local new_seq = tonumber(ARGV[2])
local max_msgs = tonumber(ARGV[3])
local ttl = tonumber(ARGV[4])

local current = redis.call('GET', key)
if current == false then
  return 0
end

local ok, data = pcall(cjson.decode, current)
if not ok or type(data) ~= 'table' then
  redis.call('DEL', key)
  return 0
end

local cur_seq = tonumber(data['latest_seq']) or 0
if new_seq <= cur_seq then
  redis.call('EXPIRE', key, ttl)
  return 0
end

local msgs = data['messages']
if type(msgs) ~= 'table' then
  redis.call('DEL', key)
  return 0
end

local msg_ok, new_msg = pcall(cjson.decode, ARGV[1])
if not msg_ok then
  return 0
end

msgs[#msgs + 1] = new_msg

local total = #msgs
if total > max_msgs then
  local trimmed = {}
  for i = total - max_msgs + 1, total do
    trimmed[#trimmed + 1] = msgs[i]
  end
  msgs = trimmed
end

data['messages'] = msgs
data['latest_seq'] = new_seq

local enc_ok, encoded = pcall(cjson.encode, data)
if not enc_ok then
  return 0
end

redis.call('SET', key, encoded, 'EX', ttl)
return 1
"""

_cas_populate_script_cache: dict[int, Any] = {}
_append_trim_script_cache: dict[int, Any] = {}


def _get_cas_populate_script(redis: Redis) -> Any:
    k = id(redis)
    if k not in _cas_populate_script_cache:
        _cas_populate_script_cache[k] = redis.register_script(_CAS_POPULATE_LUA)
    return _cas_populate_script_cache[k]


def _get_append_trim_script(redis: Redis) -> Any:
    k = id(redis)
    if k not in _append_trim_script_cache:
        _append_trim_script_cache[k] = redis.register_script(_APPEND_TRIM_LUA)
    return _append_trim_script_cache[k]


async def get_chat_tail(
    redis: Redis,
    conv_id: str,
) -> tuple[list[dict], int] | None:
    """Read cached tail; returns (messages, latest_seq) or None on miss. Refreshes TTL on hit."""
    key = _chat_tail_key(conv_id)
    raw = await redis.getex(key, ex=get_chat_tail_ttl())
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    msgs = data.get("messages")
    latest_seq = data.get("latest_seq", 0)
    if not isinstance(msgs, list):
        return None
    return msgs, int(latest_seq) if latest_seq is not None else 0


async def cas_populate_chat_tail(
    redis: Redis,
    conv_id: str,
    messages: list[dict],
    latest_seq: int,
) -> bool:
    """CAS-populate cache on read miss. Returns True if written, False if skipped."""
    payload = json.dumps({"messages": messages, "latest_seq": latest_seq}, default=str)
    script = _get_cas_populate_script(redis)
    result = await script(
        keys=[_chat_tail_key(conv_id)],
        args=[payload, str(latest_seq), str(get_chat_tail_ttl())],
    )
    return int(result) == 1


async def append_chat_tail(
    redis: Redis,
    conv_id: str,
    message_dict: dict,
    seq: int,
) -> bool:
    """Append message to tail (no-op if cache cold). Returns True if appended, False if skipped."""
    msg_json = json.dumps(message_dict, default=str)
    script = _get_append_trim_script(redis)
    result = await script(
        keys=[_chat_tail_key(conv_id)],
        args=[msg_json, str(seq), str(get_chat_tail_max_messages()), str(get_chat_tail_ttl())],
    )
    return int(result) == 1


async def invalidate_chat_tail(redis: Redis, conv_id: str) -> None:
    """Delete tail cache for conversation (for future edit/delete support)."""
    await redis.delete(_chat_tail_key(conv_id))


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
