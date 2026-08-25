"""Ingestion SSE stream: a broken stream must not be reported as a failed ingestion."""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from src.api.routers import documents as documents_router


class _FailingRedis:
    """Redis whose xread raises, standing in for a broker blip mid-ingest."""

    async def xread(self, *_args, **_kwargs):
        raise ConnectionError("redis went away")


class _StuckRedis:
    """Redis that never yields events, so the endpoint falls back to the DB status check."""

    def __init__(self) -> None:
        self.calls = 0

    async def xread(self, *_args, **_kwargs):
        self.calls += 1
        return []


def _doc(status: str = "processing", processing_error: str | None = None):
    return SimpleNamespace(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        status=status,
        processing_error=processing_error,
    )


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

    monkeypatch.setattr(documents_router, "DocumentRepository", _Repo)
    return await documents_router.ingestion_stream(
        document_id=doc.id,
        request=None,  # type: ignore[arg-type]
        session=None,  # type: ignore[arg-type]
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
