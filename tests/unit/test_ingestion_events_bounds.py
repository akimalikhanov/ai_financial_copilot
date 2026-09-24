"""Ingestion SSE event streams must be bounded, the same way chat's are.

The chat streams were fixed twice: once for having no maxlen and no TTL at all, and again
because setting the TTL only at normal completion left every abandoned stream leaking. The
ingestion streams never got either fix — they were written with a maxlen and no expiry, and
nothing else expires `ingestion:events:*`. That is the worse of the two paths to leave it
on: an ingestion worker that is killed mid-document writes no terminal event, so there is no
completion path to hang an expiry off in the first place.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest
from fakeredis import FakeAsyncRedis

from src.redis_client import (
    INGESTION_EVENTS_MAXLEN,
    add_ingestion_event,
    ingestion_stream_key,
)
from src.utils.config import get_ingest_events_ttl

_DOCUMENT_ID = "11111111-2222-3333-4444-555555555555"
_KEY = ingestion_stream_key(_DOCUMENT_ID)


@pytest.fixture
async def redis() -> AsyncGenerator[FakeAsyncRedis, None]:
    async with FakeAsyncRedis(decode_responses=True) as r:
        yield r


async def test_every_write_sets_a_ttl(redis: FakeAsyncRedis) -> None:
    """The first stage event is enough — a document killed right after it still expires."""
    await add_ingestion_event(redis, _DOCUMENT_ID, "stage", {"stage": "download_pdf"})

    ttl = await redis.ttl(_KEY)
    assert 0 < ttl <= get_ingest_events_ttl()


async def test_ttl_is_refreshed_while_the_pipeline_writes(redis: FakeAsyncRedis) -> None:
    """A stream must outlive a document that legitimately takes longer than one TTL."""
    await add_ingestion_event(redis, _DOCUMENT_ID, "stage", {"stage": "parse_pdf_docling"})
    await redis.expire(_KEY, 5)

    await add_ingestion_event(redis, _DOCUMENT_ID, "stage", {"stage": "chunk_document"})

    assert await redis.ttl(_KEY) > 5


async def test_stream_is_capped(redis: FakeAsyncRedis) -> None:
    """One event per stage, so hitting the cap means something is emitting in a loop."""
    for i in range(INGESTION_EVENTS_MAXLEN + 20):
        await add_ingestion_event(redis, _DOCUMENT_ID, "stage", {"stage": f"s{i}"})

    assert await redis.xlen(_KEY) == INGESTION_EVENTS_MAXLEN


async def test_payload_round_trips(redis: FakeAsyncRedis) -> None:
    """The subscriber parses `payload` and reads `type` out of it, as the chat stream does."""
    await add_ingestion_event(redis, _DOCUMENT_ID, "done", {"chunks": 42})

    entries = await redis.xrange(_KEY)
    assert len(entries) == 1
    assert '"type": "done"' in entries[0][1]["payload"]
    assert '"chunks": 42' in entries[0][1]["payload"]
