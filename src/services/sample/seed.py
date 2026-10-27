"""Seed the sample corpus: `python -m src.services.sample.seed`.

Reconciles the demo user's documents with the manifest through the normal ingestion task,
then waits for them to be ready. Safe to re-run; a ready corpus is a no-op. Exits 1 on an
ingestion failure or timeout, so a Job retry starts the failed document over.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast
from uuid import UUID

from celery import Task
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.api.logging import configure_logging
from src.db.connection import get_session_factory, init_db, shutdown_db
from src.models.document import Document
from src.models.user import User
from src.redis_client import close_redis_client, create_redis_app_client
from src.repository import DocumentRepository
from src.repository.user_repository import UserRepository
from src.services.ingestion.purge import purge_document
from src.services.ingestion.recovery import reap_abandoned
from src.services.ingestion.s3_client import build_raw_storage_key, upload_pdf
from src.services.ingestion.tasks import ingest_document
from src.services.router.company_name import normalize_company
from src.services.sample.manifest import SampleDocument, SampleManifest, load_manifest
from src.utils.config import (
    get_demo_user_email,
    get_demo_user_id,
    get_sample_manifest_path,
    get_sample_seed_poll_seconds,
    get_sample_seed_timeout_seconds,
)

logger = logging.getLogger(__name__)


@dataclass
class SeedPlan:
    purge: list[Document] = field(default_factory=list)
    create: list[SampleDocument] = field(default_factory=list)
    enqueue: list[UUID] = field(default_factory=list)
    # `processing` rows: re-enqueued only if their worker is gone (reap_abandoned).
    processing: list[Document] = field(default_factory=list)


def plan_seed(owned: Sequence[Document], manifest: SampleManifest) -> SeedPlan:
    """What to do with the demo user's documents. Failed and unlisted ones are purged;
    a failed one is then created again from scratch."""
    wanted = {d.id: d for d in manifest.documents}
    plan = SeedPlan()
    kept: set[UUID] = set()
    for doc in owned:
        if doc.id not in wanted or doc.status == "failed":
            plan.purge.append(doc)
            continue
        kept.add(doc.id)
        if doc.status == "pending":
            plan.enqueue.append(doc.id)
        elif doc.status == "processing":
            plan.processing.append(doc)
    plan.create = [d for d in manifest.documents if d.id not in kept]
    return plan


async def _ensure_demo_user(session: AsyncSession, demo_id: UUID) -> None:
    """The demo user owns the corpus only: no password and inactive, so it can't log in."""
    repo = UserRepository(session)
    if await repo.get_by_id(demo_id) is not None:
        return
    email = get_demo_user_email()
    if await repo.get_by_email(email) is not None:
        raise RuntimeError(f"{email} belongs to a user other than DEMO_USER_ID")
    session.add(User(id=demo_id, email=email, display_name="Sample corpus", is_active=False))
    await session.flush()


async def _create(session: AsyncSession, demo_id: UUID, doc: SampleDocument) -> None:
    data = (get_sample_manifest_path().parent / doc.file).read_bytes()
    if hashlib.sha256(data).hexdigest() != doc.sha256:
        raise RuntimeError(f"{doc.file} does not match its sha256 in the manifest")
    company = doc.metadata.get("company")
    await DocumentRepository(session).create(
        id=doc.id,
        user_id=demo_id,
        original_filename=doc.file,
        storage_key=build_raw_storage_key(demo_id, doc.id, doc.file),
        file_size_bytes=len(data),
        metadata=doc.metadata,
        company_norm=normalize_company(company) if company else None,
    )
    await session.commit()
    try:
        await upload_pdf(
            user_id=demo_id, doc_id=doc.id, filename=doc.file, fileobj=io.BytesIO(data)
        )
    except Exception:
        await session.execute(text("DELETE FROM documents WHERE id = :id"), {"id": doc.id})
        await session.commit()
        raise


async def _reconcile(
    sf: async_sessionmaker[AsyncSession], demo_id: UUID, manifest: SampleManifest
) -> None:
    async with sf() as session:
        await _ensure_demo_user(session, demo_id)
        await session.commit()
        plan = plan_seed(await DocumentRepository(session).list_by_user(demo_id), manifest)
        for doc in plan.purge:
            logger.info("sample_seed.purge", extra={"document_id": str(doc.id)})
            await purge_document(session, doc)
            await session.commit()
        for doc in plan.create:
            logger.info("sample_seed.create", extra={"document_id": str(doc.id)})
            await _create(session, demo_id, doc)
            plan.enqueue.append(doc.id)
    for doc_id in plan.enqueue:
        cast(Task, ingest_document).delay(str(doc_id))
    if plan.processing:
        redis = await create_redis_app_client()
        try:
            await reap_abandoned(redis, plan.processing, source="sample_seed")
        finally:
            await close_redis_client(redis)


async def _wait_ready(sf: async_sessionmaker[AsyncSession], ids: list[UUID]) -> bool:
    deadline = time.monotonic() + get_sample_seed_timeout_seconds()
    while True:
        async with sf() as session:
            docs = await DocumentRepository(session).get_by_ids(ids)
        failed = [d for d in docs if d.status == "failed"]
        if failed or len(docs) != len(ids):
            for d in failed:
                logger.error(
                    "sample_seed.failed",
                    extra={"document_id": str(d.id), "error": d.processing_error},
                )
            return False
        if all(d.status == "ready" for d in docs):
            return True
        if time.monotonic() > deadline:
            logger.error(
                "sample_seed.timeout", extra={"statuses": {str(d.id): d.status for d in docs}}
            )
            return False
        await asyncio.sleep(get_sample_seed_poll_seconds())


async def main() -> int:
    configure_logging()
    demo_id = get_demo_user_id()
    if demo_id is None:
        logger.info("sample_seed.disabled")
        return 0
    manifest = load_manifest()
    await init_db()
    try:
        sf = get_session_factory()
        await _reconcile(sf, demo_id, manifest)
        ok = await _wait_ready(sf, [d.id for d in manifest.documents])
    finally:
        await shutdown_db()
    logger.info("sample_seed.done", extra={"ok": ok})
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
