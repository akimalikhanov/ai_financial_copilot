"""Unit tests for chat rate limiting (Redis sliding window)."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from redis.asyncio import Redis

from src.api.deps import chat_rate_limit
from src.models.user import User
from src.redis_client import check_chat_rate_limit


def _mock_redis(result: list[Any]) -> Redis:
    """Create a mock Redis whose rate-limit script returns the given result."""
    script = AsyncMock(return_value=result)

    # Minimal object that quacks like Redis for check_chat_rate_limit
    class MockRedis:
        def register_script(self, _: str) -> Any:
            return script

    return cast(Redis, MockRedis())


@pytest.mark.asyncio
async def test_check_chat_rate_limit_allowed() -> None:
    """When Redis returns allowed=1, check_chat_rate_limit returns (True, count)."""
    redis = _mock_redis([1, 5])
    allowed, count = await check_chat_rate_limit(redis, "user-123")
    assert allowed is True
    assert count == 5


@pytest.mark.asyncio
async def test_check_chat_rate_limit_blocked() -> None:
    """When Redis returns allowed=0, check_chat_rate_limit returns (False, count)."""
    redis = _mock_redis([0, 30])
    allowed, count = await check_chat_rate_limit(redis, "user-456")
    assert allowed is False
    assert count == 30


@pytest.mark.asyncio
async def test_check_chat_rate_limit_blocked_with_decode_responses() -> None:
    """With decode_responses=True, Redis may return strings; int() handles both."""
    redis = _mock_redis(["0", "2"])
    allowed, count = await check_chat_rate_limit(redis, "user-789")
    assert allowed is False
    assert count == 2


@pytest.mark.asyncio
async def test_chat_rate_limit_passes_when_allowed() -> None:
    """chat_rate_limit returns None when under limit."""
    redis = _mock_redis([1, 1])
    user = User(
        id=UUID("a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11"),
        email="u@example.com",
        display_name=None,
        auth_provider="local",
        is_active=True,
    )
    await chat_rate_limit(redis, user)


@pytest.mark.asyncio
async def test_chat_rate_limit_raises_429_when_over_limit() -> None:
    """chat_rate_limit raises 429 with Retry-After when over limit."""
    from fastapi import HTTPException

    redis = _mock_redis([0, 30])
    user = User(
        id=UUID("b1ffcd00-0a1c-5f19-cc7e-7cc0ce491b22"),
        email="v@example.com",
        display_name=None,
        auth_provider="local",
        is_active=True,
    )
    with pytest.raises(HTTPException) as exc_info:
        await chat_rate_limit(redis, user)
    exc = exc_info.value
    assert exc.status_code == 429
    assert "Rate limit exceeded" in exc.detail
    assert exc.headers is not None and exc.headers.get("Retry-After") == "60"
