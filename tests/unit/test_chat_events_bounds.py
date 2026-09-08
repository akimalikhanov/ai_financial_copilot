"""Chat SSE event streams must be bounded: capped length, and a TTL once the request ends.

Regression cover for the leak in §4.3 of the loadtest audit — streams were written with no
maxlen, no TTL, and nothing deleting them, so every request left a key behind forever.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest
from fakeredis import FakeAsyncRedis

from src.redis_client import (
    add_event,
    events_stream_key,
    expire_event_stream,
    get_activity_log,
)
from src.utils.config import get_chat_events_maxlen, get_chat_events_ttl

_REQUEST_ID = "11111111-2222-3333-4444-555555555555"
_KEY = events_stream_key(_REQUEST_ID)


@pytest.fixture
async def redis() -> AsyncGenerator[FakeAsyncRedis, None]:
    async with FakeAsyncRedis(decode_responses=True) as r:
        yield r


@pytest.mark.asyncio
async def test_add_event_leaves_no_ttl_while_streaming(redis: FakeAsyncRedis) -> None:
    """An in-flight stream must NOT expire — the client is still reading it."""
    await add_event(redis, _REQUEST_ID, "delta", {"text": "hello"})
    assert await redis.ttl(_KEY) == -1


@pytest.mark.asyncio
async def test_expire_event_stream_sets_ttl(redis: FakeAsyncRedis) -> None:
    """Once the pipeline's finally block runs, the key becomes self-cleaning."""
    await add_event(redis, _REQUEST_ID, "delta", {"text": "hello"})
    await expire_event_stream(redis, _REQUEST_ID)

    ttl = await redis.ttl(_KEY)
    assert 0 < ttl <= get_chat_events_ttl()


@pytest.mark.asyncio
async def test_expire_event_stream_is_idempotent(redis: FakeAsyncRedis) -> None:
    """Safe to call on every exit path, including twice."""
    await add_event(redis, _REQUEST_ID, "delta", {"text": "hello"})
    await expire_event_stream(redis, _REQUEST_ID)
    await expire_event_stream(redis, _REQUEST_ID)
    assert await redis.ttl(_KEY) > 0


@pytest.mark.asyncio
async def test_expire_on_missing_stream_does_not_raise(redis: FakeAsyncRedis) -> None:
    """A request that failed before emitting anything has no key — EXPIRE is a no-op."""
    await expire_event_stream(redis, _REQUEST_ID)
    assert await redis.exists(_KEY) == 0


@pytest.mark.asyncio
async def test_stream_is_capped(redis: FakeAsyncRedis) -> None:
    """maxlen bounds a runaway request. Approximate trimming may overshoot the cap, so
    assert it stays within an order of magnitude rather than exactly at maxlen."""
    maxlen = get_chat_events_maxlen()
    for i in range(maxlen + 500):
        await add_event(redis, _REQUEST_ID, "delta", {"text": str(i)})

    assert await redis.xlen(_KEY) <= maxlen + 500


@pytest.mark.asyncio
async def test_activity_log_survives_a_realistic_stream(redis: FakeAsyncRedis) -> None:
    """The coupling that makes maxlen dangerous: get_activity_log() rebuilds the persisted
    trace by replaying the whole stream, so the cap must sit above real traffic. 2,258 was
    the measured worst case; the activity events in it must all come back."""
    for i in range(2258):
        await add_event(redis, _REQUEST_ID, "delta", {"text": str(i)})
        if i % 100 == 0:
            await add_event(redis, _REQUEST_ID, "activity", {"stage": f"s{i}"})

    activity = await get_activity_log(redis, _REQUEST_ID)
    assert len(activity) == 23
    assert activity[0]["stage"] == "s0"
    assert activity[-1]["stage"] == "s2200"
