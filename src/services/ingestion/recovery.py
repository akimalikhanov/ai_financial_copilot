"""Recovering documents whose ingestion worker died without reporting it.

A worker that is OOMKilled, evicted or node-lost runs no exception handler, so its document
stays at `processing` until the broker's visibility timeout redelivers the task — an hour,
sized by the longest legitimate parse and not shrinkable below it. The running task holds a
Redis lease it refreshes while it works, so a `processing` row with no lease is a document
whose owner is gone, and that is readable in one round trip from anywhere.

This runs on read paths rather than in a scheduler: there is no celery beat deployment, and
the moment someone is looking at a document is exactly the moment it is worth recovering.
Two API replicas reaching the same row is harmless — both enqueue, one delivery wins the
lease and the other returns before spending an attempt.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import cast
from uuid import UUID

from celery import Task
from redis.asyncio import Redis

from src.models.document import Document
from src.observability.metrics import INGESTION_REAPED
from src.redis_client import ingestion_lease_key, ingestion_reap_key
from src.utils.config import get_ingest_reap_debounce_seconds

logger = logging.getLogger(__name__)


async def reap_abandoned(
    redis: Redis,
    documents: Sequence[Document],
    *,
    source: str,
) -> set[UUID]:
    """Re-enqueue every document that claims to be processing but holds no lease.

    Returns the ids re-enqueued. Never raises: recovery failing must not fail the read it is
    riding on, and the broker's redelivery is still there underneath as the slow path.
    """
    candidates = [d for d in documents if d.status == "processing"]
    if not candidates:
        return set()

    try:
        async with redis.pipeline(transaction=False) as pipe:
            for doc in candidates:
                pipe.exists(ingestion_lease_key(str(doc.id)))
            leases = await pipe.execute()
    except Exception:
        logger.warning("ingestion.reap_check_failed", extra={"source": source}, exc_info=True)
        return set()

    reaped: set[UUID] = set()
    for doc, has_lease in zip(candidates, leases, strict=False):
        if has_lease:
            continue
        if await _enqueue_once(redis, doc.id, source):
            reaped.add(doc.id)
    return reaped


async def _enqueue_once(redis: Redis, document_id: UUID, source: str) -> bool:
    """Re-enqueue one document unless it was already re-enqueued recently."""
    from src.services.ingestion.tasks import ingest_document

    reap_key = ingestion_reap_key(str(document_id))
    try:
        claimed = await redis.set(reap_key, source, nx=True, ex=get_ingest_reap_debounce_seconds())
    except Exception:
        logger.warning(
            "ingestion.reap_debounce_failed",
            extra={"document_id": str(document_id), "source": source},
            exc_info=True,
        )
        return False
    if not claimed:
        return False

    try:
        # Celery's producer is synchronous and connects on first use, so a broker that is
        # slow to answer would otherwise stall the event loop for every request this worker
        # is serving, not just this one.
        await asyncio.to_thread(cast(Task, ingest_document).delay, str(document_id))
    except Exception:
        # Drop the debounce so the next read tries again rather than waiting it out.
        await redis.delete(reap_key)
        logger.exception(
            "ingestion.reap_enqueue_failed",
            extra={"document_id": str(document_id), "source": source},
        )
        return False

    INGESTION_REAPED.labels(source).inc()
    logger.warning(
        "ingestion.reaped",
        extra={"document_id": str(document_id), "source": source},
    )
    return True
