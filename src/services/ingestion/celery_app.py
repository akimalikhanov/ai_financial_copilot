"""Celery app and tasks for PDF ingestion. API enqueues; worker updates document status."""

from __future__ import annotations

import asyncio
import logging
from uuid import UUID

from celery import Celery
from celery.exceptions import Retry
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from src.repository.document_repository import DocumentRepository
from src.utils.config import get_db_url, get_redis_broker_url

logger = logging.getLogger(__name__)

celery_app = Celery(
    "ingestion",
    broker=get_redis_broker_url(),
    include=["src.services.ingestion.celery_app"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    worker_hijack_root_logger=False,  # Use our JSON logging from configure_worker_logging
)


async def _set_document_ready(document_id: str) -> bool:
    engine = create_async_engine(get_db_url(), poolclass=NullPool)
    async with async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
    )() as session:
        repo = DocumentRepository(session)
        updated = await repo.update_status(UUID(document_id), "ready")
        await session.commit()
    await engine.dispose()
    return updated


@celery_app.task(bind=True, name="ingest_document")
def ingest_document(self, document_id: str) -> None:
    """Set document status to ready. Enqueued by upload API."""
    logger.info("ingest_document.received", extra={"document_id": document_id})
    try:
        updated = asyncio.run(_set_document_ready(document_id))
        if not updated:
            # Usually means the API transaction hasn't committed yet.
            logger.warning(
                "ingest_document.not_found_retrying",
                extra={"document_id": document_id, "attempt": getattr(self.request, "retries", 0)},
            )
            raise self.retry(countdown=1, max_retries=10)
        logger.info("ingest_document.done", extra={"document_id": document_id})
    except Retry:
        raise
    except Exception:
        logger.exception("ingest_document.failed", extra={"document_id": document_id})
        raise
