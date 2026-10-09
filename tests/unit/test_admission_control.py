"""Unit tests for global chat admission control (broker queue depth -> 503)."""

from __future__ import annotations

from typing import cast

import pytest
from fastapi import HTTPException
from redis.asyncio import Redis

from src.api.deps import CHAT_BROKER_QUEUE_KEY, chat_admission_control


def _mock_broker(depth: int) -> tuple[Redis, list[str]]:
    """Mock broker Redis returning a fixed LLEN, recording the key it was asked about."""
    seen: list[str] = []

    class MockRedis:
        async def llen(self, key: str) -> int:
            seen.append(key)
            return depth

    return cast(Redis, MockRedis()), seen


@pytest.mark.asyncio
async def test_admits_when_queue_below_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHAT_QUEUE_MAX_DEPTH", "36")
    redis, seen = _mock_broker(35)
    await chat_admission_control(redis)
    assert seen == [CHAT_BROKER_QUEUE_KEY]


@pytest.mark.asyncio
async def test_rejects_at_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    """The threshold is inclusive: at the cap the queue is already as deep as promised."""
    monkeypatch.setenv("CHAT_QUEUE_MAX_DEPTH", "36")
    redis, _ = _mock_broker(36)
    with pytest.raises(HTTPException) as exc_info:
        await chat_admission_control(redis)
    exc = exc_info.value
    assert exc.status_code == 503
    assert "at capacity" in exc.detail
    assert exc.headers is not None and exc.headers.get("Retry-After") == "30"


@pytest.mark.asyncio
async def test_threshold_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A depth that rejects at the default must admit once the cap is raised."""
    monkeypatch.setenv("CHAT_QUEUE_MAX_DEPTH", "100")
    redis, _ = _mock_broker(40)
    await chat_admission_control(redis)


@pytest.mark.asyncio
async def test_rejection_increments_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.observability.metrics import CHAT_ADMISSION_REJECTED

    monkeypatch.setenv("CHAT_QUEUE_MAX_DEPTH", "1")
    before = CHAT_ADMISSION_REJECTED._value.get()
    redis, _ = _mock_broker(5)
    with pytest.raises(HTTPException):
        await chat_admission_control(redis)
    assert CHAT_ADMISSION_REJECTED._value.get() == before + 1
