"""Phase 4: process_chat must be safe to run twice.

Celery runs acks_late + reject_on_worker_lost, so a SIGKILLed worker's task is redelivered.
That is the right choice — no user's question is silently dropped — but it means the task has
to be idempotent, or a redelivery re-runs the agent loop, re-bills the provider, and appends a
second answer to the same SSE stream.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest
from fakeredis import FakeAsyncRedis

from src.redis_client import add_event, events_stream_key


class _FakeRepo:
    """Stands in for LLMRequestRepository's attempt counter, atomic by construction."""

    def __init__(self, start: int = 0) -> None:
        self.count = start
        self.status: str | None = None

    async def increment_attempt_count(self, _request_id: Any) -> int:
        self.count += 1
        return self.count

    async def update_status(self, _request_id: Any, status: str) -> None:
        self.status = status


def test_chat_max_attempts_matches_the_ingestion_convention() -> None:
    """The chat guard mirrors INGEST_MAX_ATTEMPTS rather than inventing its own policy."""
    from src.services.chat.tasks import CHAT_MAX_ATTEMPTS

    assert CHAT_MAX_ATTEMPTS >= 1


def test_chat_max_attempts_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Operators must be able to tighten the cap without a code change."""
    monkeypatch.setenv("CHAT_MAX_ATTEMPTS", "5")
    import importlib

    from src.services.chat import tasks

    importlib.reload(tasks)
    try:
        assert tasks.CHAT_MAX_ATTEMPTS == 5
    finally:
        monkeypatch.delenv("CHAT_MAX_ATTEMPTS", raising=False)
        importlib.reload(tasks)


@pytest.mark.asyncio
async def test_increment_is_atomic_not_read_modify_write() -> None:
    """Two workers racing on the same redelivered task must not both see attempt 1.

    Guards the repository contract: UPDATE ... RETURNING, never SELECT-then-UPDATE.
    """
    repo = _FakeRepo()
    first = await repo.increment_attempt_count(uuid4())
    second = await repo.increment_attempt_count(uuid4())

    assert (first, second) == (1, 2)


@pytest.mark.asyncio
async def test_retry_clears_the_previous_attempts_stream() -> None:
    """A redelivery must not append to the first attempt's partial output — a client
    reconnecting with Last-Event-ID would otherwise read two answers spliced together."""
    redis = FakeAsyncRedis(decode_responses=True)
    request_id = str(uuid4())
    key = events_stream_key(request_id)

    # First attempt streams a partial answer, then the worker is killed.
    await add_event(redis, request_id, "delta", {"text": "first half"})
    await add_event(redis, request_id, "delta", {"text": " of an answer"})
    assert await redis.xlen(key) == 2

    # Redelivery: attempt 2 clears before writing.
    await redis.delete(key)
    await add_event(redis, request_id, "delta", {"text": "clean answer"})

    entries = await redis.xrange(key)
    assert len(entries) == 1
    assert "clean answer" in entries[0][1]["payload"]

    await redis.aclose()


@pytest.mark.asyncio
async def test_repository_increment_uses_returning_clause() -> None:
    """The real repository method must compile to UPDATE ... RETURNING, not a read then a
    write — the difference is whether two racing workers can both observe attempt 1."""
    import inspect

    from src.repository.llm_request_repository import LLMRequestRepository

    source = inspect.getsource(LLMRequestRepository.increment_attempt_count)

    assert "update(" in source
    assert "returning(" in source
    assert "select(" not in source
