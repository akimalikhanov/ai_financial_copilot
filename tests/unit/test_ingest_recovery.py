"""Re-enqueueing documents whose ingestion worker died without reporting it.

The lease is what makes this decidable: a row at `processing` that holds no lease has no
live owner, because the task claims the lease before it flips the row. Everything here is
about not over-reacting to that — a queued document looks the same from the outside.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Any

import pytest
from fakeredis import FakeAsyncRedis

from src.redis_client import ingestion_lease_key, ingestion_reap_key
from src.services.ingestion import recovery, tasks


@pytest.fixture
def redis() -> FakeAsyncRedis:
    return FakeAsyncRedis(decode_responses=True)


@pytest.fixture
def enqueued(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Captures what would have been put on the ingestion queue."""
    sent: list[str] = []
    monkeypatch.setattr(tasks.ingest_document, "delay", lambda doc_id: sent.append(doc_id))
    return sent


def _doc(status: str = "processing") -> Any:
    return SimpleNamespace(id=uuid.uuid4(), status=status)


class TestReap:
    async def test_processing_without_a_lease_is_re_enqueued(
        self, redis: FakeAsyncRedis, enqueued: list[str]
    ) -> None:
        doc = _doc()

        reaped = await recovery.reap_abandoned(redis, [doc], source="list")

        assert reaped == {doc.id}
        assert enqueued == [str(doc.id)]

    async def test_a_live_lease_is_left_alone(
        self, redis: FakeAsyncRedis, enqueued: list[str]
    ) -> None:
        """The document is mid-parse. Re-enqueueing it would be refused by the running task,
        but it would still cost a queue slot on a worker with one of them."""
        doc = _doc()
        await redis.set(ingestion_lease_key(str(doc.id)), "token", ex=45)

        assert await recovery.reap_abandoned(redis, [doc], source="list") == set()
        assert enqueued == []

    @pytest.mark.parametrize("status", ["pending", "ready", "failed"])
    async def test_only_processing_rows_are_candidates(
        self, redis: FakeAsyncRedis, enqueued: list[str], status: str
    ) -> None:
        """`pending` is the one that would bite: a queued document holds no lease either, and
        re-enqueueing it would double every upload."""
        assert await recovery.reap_abandoned(redis, [_doc(status)], source="list") == set()
        assert enqueued == []

    async def test_a_reaped_document_is_not_re_enqueued_again(
        self, redis: FakeAsyncRedis, enqueued: list[str]
    ) -> None:
        """It stays `processing` with no lease for as long as it sits in the queue, so every
        read in between would otherwise queue another copy of it.

        Measured in a T10b re-run before this marker was scoped to the in-flight enqueue: a
        document queued behind two large filings was re-enqueued once per marker TTL.
        """
        doc = _doc()

        first = await recovery.reap_abandoned(redis, [doc], source="list")
        second = await recovery.reap_abandoned(redis, [doc], source="stream")
        third = await recovery.reap_abandoned(redis, [doc], source="list")

        assert (first, second, third) == ({doc.id}, set(), set())
        assert enqueued == [str(doc.id)]

    async def test_the_marker_lifts_once_a_worker_takes_the_document(
        self, redis: FakeAsyncRedis, enqueued: list[str]
    ) -> None:
        """The task deletes the marker when it claims the lease. A worker that then dies is
        immediately reapable again rather than waiting out a window it no longer needs."""
        doc = _doc()
        await recovery.reap_abandoned(redis, [doc], source="list")

        # What the pipeline does on a successful claim.
        await redis.delete(ingestion_reap_key(str(doc.id)))

        assert await recovery.reap_abandoned(redis, [doc], source="list") == {doc.id}
        assert len(enqueued) == 2

    async def test_a_failed_enqueue_clears_the_debounce(
        self, redis: FakeAsyncRedis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Otherwise a broker blip would silence recovery for the whole debounce window."""
        doc = _doc()

        def _boom(_doc_id: str) -> None:
            raise ConnectionError("broker unreachable")

        monkeypatch.setattr(tasks.ingest_document, "delay", _boom)

        assert await recovery.reap_abandoned(redis, [doc], source="list") == set()
        assert await redis.exists(ingestion_reap_key(str(doc.id))) == 0

    async def test_redis_failure_does_not_break_the_read(self, enqueued: list[str]) -> None:
        """This rides on `list_documents` and on the SSE loop. Neither may fail because
        recovery could not run — the broker's redelivery is still underneath it."""

        class _BrokenRedis:
            def pipeline(self, *_args: object, **_kwargs: object) -> object:
                raise ConnectionError("redis went away")

        assert await recovery.reap_abandoned(_BrokenRedis(), [_doc()], source="list") == set()  # type: ignore[arg-type]
        assert enqueued == []

    async def test_no_processing_rows_costs_no_round_trip(self) -> None:
        """The common case: every document is ready, so the list endpoint touches Redis not
        at all."""

        class _UnusableRedis:
            def pipeline(self, *_args: object, **_kwargs: object) -> object:
                raise AssertionError("must not touch Redis when nothing is processing")

        docs = [_doc("ready"), _doc("failed")]

        assert await recovery.reap_abandoned(_UnusableRedis(), docs, source="list") == set()  # type: ignore[arg-type]
