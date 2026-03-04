"""Unit tests for chat tail cache (Lua scripts + get_chat_tail)."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator

import pytest
from fakeredis import FakeAsyncRedis

from src.redis_client import append_chat_tail, cas_populate_chat_tail, get_chat_tail
from src.utils.config import get_chat_tail_max_messages, get_chat_tail_ttl


@pytest.fixture
async def redis() -> AsyncGenerator[FakeAsyncRedis, None]:
    async with FakeAsyncRedis() as r:
        yield r


# --- CAS-Populate ---


@pytest.mark.asyncio
async def test_cas_populate_empty_key_writes(redis: FakeAsyncRedis) -> None:
    """Empty key: writes and returns 1."""
    payload = {"messages": [{"role": "user", "content": "hi"}], "latest_seq": 1}
    result = await cas_populate_chat_tail(redis, "conv-1", payload["messages"], 1)
    assert result is True
    raw = await redis.get("chat:tail:conv-1")
    assert raw is not None
    data = json.loads(raw)
    assert data["messages"] == payload["messages"]
    assert data["latest_seq"] == 1


@pytest.mark.asyncio
async def test_cas_populate_lower_seq_overwrites(redis: FakeAsyncRedis) -> None:
    """Existing cache with lower latest_seq: overwrites, returns 1."""
    old = json.dumps({"messages": [{"role": "user", "content": "old"}], "latest_seq": 1})
    await redis.set("chat:tail:conv-1", old, ex=get_chat_tail_ttl())
    new_msgs = [{"role": "user", "content": "new"}]
    result = await cas_populate_chat_tail(redis, "conv-1", new_msgs, 2)
    assert result is True
    raw = await redis.get("chat:tail:conv-1")
    data = json.loads(raw)
    assert data["messages"] == new_msgs
    assert data["latest_seq"] == 2


@pytest.mark.asyncio
async def test_cas_populate_equal_seq_skips(redis: FakeAsyncRedis) -> None:
    """Existing cache with equal latest_seq: skips, refreshes TTL, returns 0."""
    old = json.dumps({"messages": [{"role": "user", "content": "keep"}], "latest_seq": 2})
    await redis.set("chat:tail:conv-1", old, ex=get_chat_tail_ttl())
    result = await cas_populate_chat_tail(redis, "conv-1", [{"role": "user", "content": "new"}], 2)
    assert result is False
    raw = await redis.get("chat:tail:conv-1")
    data = json.loads(raw)
    assert data["messages"][0]["content"] == "keep"


@pytest.mark.asyncio
async def test_cas_populate_higher_seq_skips(redis: FakeAsyncRedis) -> None:
    """Existing cache with higher latest_seq: skips, refreshes TTL, returns 0."""
    old = json.dumps({"messages": [{"role": "user", "content": "keep"}], "latest_seq": 3})
    await redis.set("chat:tail:conv-1", old, ex=get_chat_tail_ttl())
    result = await cas_populate_chat_tail(redis, "conv-1", [{"role": "user", "content": "new"}], 2)
    assert result is False
    raw = await redis.get("chat:tail:conv-1")
    data = json.loads(raw)
    assert data["messages"][0]["content"] == "keep"
    assert data["latest_seq"] == 3


@pytest.mark.asyncio
async def test_cas_populate_corrupt_json_overwrites(redis: FakeAsyncRedis) -> None:
    """Corrupt JSON in existing cache: overwrites, returns 1."""
    await redis.set("chat:tail:conv-1", "not valid json", ex=get_chat_tail_ttl())
    new_msgs = [{"role": "user", "content": "new"}]
    result = await cas_populate_chat_tail(redis, "conv-1", new_msgs, 1)
    assert result is True
    raw = await redis.get("chat:tail:conv-1")
    data = json.loads(raw)
    assert data["messages"] == new_msgs
    assert data["latest_seq"] == 1


# --- Append-Trim-Expire ---


@pytest.mark.asyncio
async def test_append_key_missing_returns_zero(redis: FakeAsyncRedis) -> None:
    """Key does not exist: returns 0 (no-op, does not create)."""
    result = await append_chat_tail(redis, "conv-1", {"role": "user", "content": "hi"}, 1)
    assert result is False
    assert await redis.get("chat:tail:conv-1") is None


@pytest.mark.asyncio
async def test_append_higher_seq_appends(redis: FakeAsyncRedis) -> None:
    """Append with seq higher than cached: appends, returns 1."""
    data = json.dumps({"messages": [{"role": "user", "content": "hi"}], "latest_seq": 1})
    await redis.set("chat:tail:conv-1", data, ex=get_chat_tail_ttl())
    result = await append_chat_tail(redis, "conv-1", {"role": "assistant", "content": "hello"}, 2)
    assert result is True
    raw = await redis.get("chat:tail:conv-1")
    parsed = json.loads(raw)
    assert len(parsed["messages"]) == 2
    assert parsed["messages"][1]["content"] == "hello"
    assert parsed["latest_seq"] == 2


@pytest.mark.asyncio
async def test_append_equal_seq_skips(redis: FakeAsyncRedis) -> None:
    """Append with seq equal to cached: skips, refreshes TTL, returns 0."""
    data = json.dumps({"messages": [{"role": "user", "content": "hi"}], "latest_seq": 2})
    await redis.set("chat:tail:conv-1", data, ex=get_chat_tail_ttl())
    result = await append_chat_tail(redis, "conv-1", {"role": "assistant", "content": "dup"}, 2)
    assert result is False
    raw = await redis.get("chat:tail:conv-1")
    parsed = json.loads(raw)
    assert len(parsed["messages"]) == 1


@pytest.mark.asyncio
async def test_append_lower_seq_skips(redis: FakeAsyncRedis) -> None:
    """Append with seq lower than cached: skips, refreshes TTL, returns 0."""
    data = json.dumps({"messages": [{"role": "user", "content": "hi"}], "latest_seq": 3})
    await redis.set("chat:tail:conv-1", data, ex=get_chat_tail_ttl())
    result = await append_chat_tail(redis, "conv-1", {"role": "assistant", "content": "old"}, 2)
    assert result is False
    raw = await redis.get("chat:tail:conv-1")
    parsed = json.loads(raw)
    assert len(parsed["messages"]) == 1
    assert parsed["latest_seq"] == 3


@pytest.mark.asyncio
async def test_append_trim_drops_oldest(redis: FakeAsyncRedis) -> None:
    """Trim: pre-populate with 50 messages, append one more, verify only 50 remain."""
    msgs = [{"role": "user", "content": f"msg-{i}"} for i in range(get_chat_tail_max_messages())]
    data = json.dumps({"messages": msgs, "latest_seq": get_chat_tail_max_messages()})
    await redis.set("chat:tail:conv-1", data, ex=get_chat_tail_ttl())
    result = await append_chat_tail(
        redis,
        "conv-1",
        {"role": "assistant", "content": "new"},
        get_chat_tail_max_messages() + 1,
    )
    assert result is True
    raw = await redis.get("chat:tail:conv-1")
    parsed = json.loads(raw)
    assert len(parsed["messages"]) == get_chat_tail_max_messages()
    assert parsed["messages"][0]["content"] == "msg-1"  # msg-0 dropped
    assert parsed["messages"][-1]["content"] == "new"


@pytest.mark.asyncio
async def test_append_corrupt_json_dels_key(redis: FakeAsyncRedis) -> None:
    """Corrupt JSON in cache: DELs key, returns 0."""
    await redis.set("chat:tail:conv-1", "corrupt", ex=get_chat_tail_ttl())
    result = await append_chat_tail(redis, "conv-1", {"role": "user", "content": "hi"}, 1)
    assert result is False
    assert await redis.get("chat:tail:conv-1") is None


# --- get_chat_tail ---


@pytest.mark.asyncio
async def test_get_chat_tail_miss_returns_none(redis: FakeAsyncRedis) -> None:
    """Key does not exist: returns None."""
    result = await get_chat_tail(redis, "conv-1")
    assert result is None


@pytest.mark.asyncio
async def test_get_chat_tail_valid_returns_messages_and_seq(redis: FakeAsyncRedis) -> None:
    """Valid data: returns (messages, latest_seq), TTL refreshed."""
    msgs = [{"role": "user", "content": "hi"}]
    data = json.dumps({"messages": msgs, "latest_seq": 1})
    await redis.set("chat:tail:conv-1", data, ex=get_chat_tail_ttl())
    result = await get_chat_tail(redis, "conv-1")
    assert result is not None
    got_msgs, got_seq = result
    assert got_msgs == msgs
    assert got_seq == 1
    ttl = await redis.ttl("chat:tail:conv-1")
    assert ttl > 0 and ttl <= get_chat_tail_ttl()


@pytest.mark.asyncio
async def test_get_chat_tail_corrupt_returns_none(redis: FakeAsyncRedis) -> None:
    """Corrupt JSON: returns None."""
    await redis.set("chat:tail:conv-1", "not json", ex=get_chat_tail_ttl())
    result = await get_chat_tail(redis, "conv-1")
    assert result is None
