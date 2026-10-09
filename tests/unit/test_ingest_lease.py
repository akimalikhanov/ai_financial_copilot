"""The ingestion lease: one Redis key that is both mutual exclusion and a liveness signal.

A worker killed without warning (OOMKill, eviction, node loss) runs no exception handler, so
the document row stays at `processing` and nothing on the read side can tell it apart from a
document that is genuinely being parsed. The lease closes that gap: the task claims the
document before it spends an attempt on it, refreshes the claim while it works, and drops it
on the way out. `processing` with no lease therefore means the owner is dead.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest
from fakeredis import FakeAsyncRedis

from src.redis_client import ingestion_lease_key
from src.services.ingestion import tasks
from src.services.ingestion.tasks import _DocumentLease

DOC_ID = "11111111-1111-1111-1111-111111111111"


@pytest.fixture
def redis() -> FakeAsyncRedis:
    return FakeAsyncRedis(decode_responses=True)


@pytest.fixture
def fast_heartbeat(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refresh on every event-loop turn, so a test can drive ticks with `asyncio.sleep(0)`."""
    monkeypatch.setenv("INGEST_HEARTBEAT_INTERVAL_SECONDS", "0")


class TestClaim:
    async def test_acquire_writes_the_key_with_a_ttl(self, redis: FakeAsyncRedis) -> None:
        """The TTL is the whole mechanism — a lease without one never expires on a dead worker."""
        lease = _DocumentLease(redis, DOC_ID)

        assert await lease.acquire() is True
        assert lease.held is True
        assert await redis.ttl(ingestion_lease_key(DOC_ID)) > 0

    async def test_second_worker_is_refused(self, redis: FakeAsyncRedis) -> None:
        """Two deliveries of one document must not both run: they would race each other
        through delete_by_document and index two generations of chunks."""
        first = _DocumentLease(redis, DOC_ID)
        second = _DocumentLease(redis, DOC_ID)

        assert await first.acquire() is True
        assert await second.acquire() is False
        assert second.held is False

    async def test_released_document_can_be_claimed_again(self, redis: FakeAsyncRedis) -> None:
        first = _DocumentLease(redis, DOC_ID)
        await first.acquire()
        await first.release()

        assert await redis.exists(ingestion_lease_key(DOC_ID)) == 0
        assert await _DocumentLease(redis, DOC_ID).acquire() is True

    async def test_release_leaves_another_workers_lease_alone(self, redis: FakeAsyncRedis) -> None:
        """Compare-and-set on release. A task whose lease expired and was re-claimed must not
        delete the new owner's key on its way out."""
        lease = _DocumentLease(redis, DOC_ID)
        await lease.acquire()
        await redis.set(ingestion_lease_key(DOC_ID), "another-workers-token")

        await lease.release()

        assert await redis.get(ingestion_lease_key(DOC_ID)) == "another-workers-token"

    async def test_redis_down_does_not_block_ingestion(self) -> None:
        """Fail open: refusing to ingest is worse than ingesting without the guard, and the
        recovery that reads the lease cannot run while Redis is down either."""
        lease = _DocumentLease(None, DOC_ID)

        assert await lease.acquire() is True
        assert lease.held is False
        await lease.release()


@pytest.mark.usefixtures("fast_heartbeat")
class TestHeartbeat:
    async def test_refresh_extends_the_ttl(self, redis: FakeAsyncRedis) -> None:
        """A parse legitimately runs far longer than the TTL, so the lease has to be kept
        alive by something that stops when the process does."""
        lease = _DocumentLease(redis, DOC_ID)
        await lease.acquire()
        key = ingestion_lease_key(DOC_ID)
        await redis.expire(key, 5)

        lease.start_heartbeat()
        for _ in range(10):
            await asyncio.sleep(0)
        ttl = await redis.ttl(key)
        await lease.release()

        assert ttl > 5

    async def test_lost_lease_cancels_the_owner(self, redis: FakeAsyncRedis) -> None:
        """If the key has become someone else's, a second worker is already on this document.
        The heartbeat cancels the task it belongs to rather than let the two run side by side."""
        lease = _DocumentLease(redis, DOC_ID)
        await lease.acquire()

        async def _work() -> str:
            lease.start_heartbeat()
            await redis.set(ingestion_lease_key(DOC_ID), "another-workers-token")
            try:
                await asyncio.sleep(30)
            finally:
                await lease.release()
            return "finished"

        with pytest.raises(asyncio.CancelledError):
            await asyncio.create_task(_work())

        assert lease.lost is True
        assert lease.held is False
        assert await redis.get(ingestion_lease_key(DOC_ID)) == "another-workers-token"

    async def test_release_stops_the_heartbeat(self, redis: FakeAsyncRedis) -> None:
        """A heartbeat outliving its task would keep a finished document's key alive, and
        every leaked coroutine holds the pipeline's locals with it."""
        lease = _DocumentLease(redis, DOC_ID)
        await lease.acquire()
        lease.start_heartbeat()
        await asyncio.sleep(0)

        await lease.release()

        assert lease._heartbeat_task is None
        assert await redis.exists(ingestion_lease_key(DOC_ID)) == 0


class TestPipelineOrdering:
    """The guards are only worth as much as their position in `_run_pipeline`."""

    def _source(self) -> str:
        return inspect.getsource(tasks._run_pipeline)

    def test_terminal_return_precedes_the_attempt_increment(self) -> None:
        """An hour-late redelivery of a document that has since succeeded would otherwise
        spend an attempt, trip the max-attempts branch, and fail a healthy row."""
        source = self._source()

        assert source.index('doc.status == "ready"') < source.index("increment_attempt_count")

    def test_claim_precedes_the_attempt_increment(self) -> None:
        """A refused delivery did no work, so it must not consume one of the attempts."""
        source = self._source()

        assert source.index("lease.acquire()") < source.index("increment_attempt_count")

    def test_claim_precedes_the_flip_to_processing(self) -> None:
        """Otherwise a row could be `processing` with no lease for reasons other than a dead
        worker, and the read side would re-enqueue a document that is running fine."""
        source = self._source()

        assert source.index("lease.acquire()") < source.index('"processing"')

    def test_lease_is_released_on_every_exit_path(self) -> None:
        source = self._source()
        finally_block = source.rsplit("\n    finally:", 1)[1]

        assert "await lease.release()" in finally_block


class TestReapMarker:
    """The claim also lifts the marker that says a re-enqueue for this document is in flight."""

    def test_the_claim_clears_the_reap_marker(self) -> None:
        source = inspect.getsource(tasks._run_pipeline)

        assert "ingestion_reap_key(document_id)" in source
        assert source.index("lease.start_heartbeat()") < source.index("ingestion_reap_key")
        assert source.index("ingestion_reap_key") < source.index("increment_attempt_count")
