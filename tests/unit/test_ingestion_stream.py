"""Ingestion SSE stream: a broken stream must not be reported as a failed ingestion.

It must also not heartbeat forever when the worker behind it is gone, which is the case
`ready`/`failed` cannot cover — a killed process writes neither. The stream re-enqueues the
document instead and stays open for the replacement attempt.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from fakeredis import FakeAsyncRedis

from src.api.routers import documents as documents_router
from src.redis_client import ingestion_lease_key
from src.services.ingestion import tasks


class _FailingRedis:
    """Redis whose xread raises, standing in for a broker blip mid-ingest."""

    async def xread(self, *_args, **_kwargs):
        raise ConnectionError("redis went away")


class _StuckRedis:
    """Redis that never yields events, so the endpoint falls back to the DB status check.

    Everything other than `xread` is a real fakeredis, because the fallback now also reads
    the document's lease.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.inner = FakeAsyncRedis(decode_responses=True)

    async def xread(self, *_args, **_kwargs):
        self.calls += 1
        return []

    def __getattr__(self, name):
        return getattr(self.inner, name)


@pytest.fixture
def enqueued(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    sent: list[str] = []
    monkeypatch.setattr(tasks.ingest_document, "delay", lambda doc_id: sent.append(doc_id))
    return sent


def _doc(
    status: str = "processing",
    processing_error: str | None = None,
    *,
    age_seconds: float = 5.0,
    attempts: int = 1,
):
    return SimpleNamespace(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        status=status,
        processing_error=processing_error,
        created_at=datetime.now(UTC) - timedelta(seconds=age_seconds),
        ingest_attempt_count=attempts,
    )


class _FakeSession:
    """Stands in for the request-scoped session, which is committed before streaming starts."""

    async def commit(self) -> None: ...


async def _collect(response, limit: int = 12) -> list[str]:
    chunks: list[str] = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)
        if len(chunks) >= limit:
            break
    return chunks


async def _call_stream(monkeypatch, doc, redis) -> object:
    class _Repo:
        def __init__(self, _session) -> None: ...

        async def get_by_id(self, _document_id):
            return doc

    class _SessionCtx:
        async def __aenter__(self):
            return None

        async def __aexit__(self, *_exc):
            return False

    monkeypatch.setattr(documents_router, "DocumentRepository", _Repo)
    monkeypatch.setattr(documents_router, "get_session_factory", lambda: _SessionCtx)
    return await documents_router.ingestion_stream(
        document_id=doc.id,
        request=None,  # type: ignore[arg-type]
        session=_FakeSession(),  # type: ignore[arg-type]
        redis=redis,  # type: ignore[arg-type]
        current_user=SimpleNamespace(id=doc.user_id),  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_redis_failure_does_not_emit_ingestion_error(monkeypatch):
    """A Redis read failure ends the stream silently — the worker is still ingesting.

    Emitting `error` here is what made the UI show a healthy in-flight document as failed.
    """
    response = await _call_stream(monkeypatch, _doc(), _FailingRedis())
    body = "".join(await _collect(response))

    assert "event: error" not in body
    assert "Stream read failed" not in body


@pytest.mark.asyncio
async def test_terminal_failed_status_still_emits_error(monkeypatch):
    """A genuinely failed document must still surface as an error event."""
    doc = _doc(status="failed", processing_error="docling exploded")
    response = await _call_stream(monkeypatch, doc, _StuckRedis())
    body = "".join(await _collect(response))

    assert "event: error" in body
    assert "docling exploded" in body


@pytest.mark.asyncio
async def test_ready_status_emits_done(monkeypatch):
    """A document that finished while the client was disconnected replays `done` on reconnect."""
    response = await _call_stream(monkeypatch, _doc(status="ready"), _StuckRedis())
    body = "".join(await _collect(response))

    assert "event: done" in body
    assert "event: error" not in body


@pytest.mark.asyncio
async def test_dead_worker_is_re_enqueued_and_the_stream_stays_open(monkeypatch, enqueued):
    """The case neither terminal status covers: the worker was killed, so nothing will ever
    write `ready` or `failed` and the uploader would heartbeat until it gave up."""
    doc = _doc()
    response = await _call_stream(monkeypatch, doc, _StuckRedis())
    body = "".join(await _collect(response))

    assert "event: retrying" in body
    assert '"attempt": 2' in body
    assert enqueued == [str(doc.id)]
    # Not terminal — `done` and `error` stay the only two events that end a stream.
    assert "event: error" not in body
    assert "event: done" not in body


@pytest.mark.asyncio
async def test_a_live_worker_is_not_disturbed(monkeypatch, enqueued):
    """A slow stage is silent for minutes at a time; a lease means the worker is still there."""
    doc = _doc()
    redis = _StuckRedis()
    await redis.inner.set(ingestion_lease_key(str(doc.id)), "token", ex=45)

    response = await _call_stream(monkeypatch, doc, redis)
    body = "".join(await _collect(response))

    assert enqueued == []
    assert "event: retrying" not in body
    assert ": keepalive" in body


@pytest.mark.asyncio
@pytest.mark.usefixtures("enqueued")
async def test_backstop_gives_up_when_the_lease_cannot_save_it(monkeypatch):
    """Last resort, for a document that has been claimed longer than any task may run — a
    lease that never expires because nothing is expiring it means Redis itself is the
    problem, and the client should stop waiting rather than heartbeat forever."""
    doc = _doc(age_seconds=3000)
    redis = _StuckRedis()
    await redis.inner.set(ingestion_lease_key(str(doc.id)), "token", ex=45)

    response = await _call_stream(monkeypatch, doc, redis)
    body = "".join(await _collect(response))

    assert "event: error" in body
    assert "stopped unexpectedly" in body
