"""Remove a document from the search indexes, S3 and Postgres."""

from __future__ import annotations

import asyncio
import logging
import os

import aioboto3
from botocore.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.document import Document
from src.services.ingestion import opensearch_ingest, qdrant_ingest
from src.utils.config import (
    get_s3_access_key,
    get_s3_chunks_bucket,
    get_s3_docling_bucket,
    get_s3_endpoint_url,
    get_s3_pictures_bucket,
    get_s3_raw_bucket,
    get_s3_rendered_bucket,
    get_s3_secret_key,
)

logger = logging.getLogger(__name__)


class IndexCleanupError(Exception):
    """Index cleanup failed; nothing else was touched, so the purge can be retried."""


async def purge_document(session: AsyncSession, doc: Document) -> None:
    """Delete the document everywhere. The caller commits."""
    document_id = doc.id
    qdrant_collection = os.getenv("QDRANT_COLLECTION", "documents")
    opensearch_index = os.getenv("OPENSEARCH_INDEX", "chunks")

    # Index cleanup must succeed before the Postgres row goes: a swallowed failure here
    # (backend down/restarting) leaves chunks indexed with no row to hydrate them, and
    # nothing ever collects them. Failing keeps the document deletable later.
    try:
        await asyncio.gather(
            asyncio.to_thread(qdrant_ingest.delete_by_document, qdrant_collection, document_id),
            asyncio.to_thread(opensearch_ingest.delete_by_document, opensearch_index, document_id),
        )
    except Exception as exc:
        logger.exception(
            "delete_document.index_cleanup_failed", extra={"document_id": str(document_id)}
        )
        raise IndexCleanupError(str(document_id)) from exc

    s3_keys = {
        get_s3_raw_bucket(): [doc.storage_key],
        get_s3_docling_bucket(): [f"processed/{doc.user_id}/{document_id}/docling.json"],
        get_s3_rendered_bucket(): [f"processed/{doc.user_id}/{document_id}/document.md"],
        get_s3_chunks_bucket(): [f"processed/{doc.user_id}/{document_id}/chunks.jsonl"],
    }
    s3_session = aioboto3.Session()
    async with s3_session.client(  # type: ignore[attr-defined]
        "s3",
        endpoint_url=get_s3_endpoint_url(),
        region_name="garage",
        aws_access_key_id=get_s3_access_key(),
        aws_secret_access_key=get_s3_secret_key(),
        config=Config(response_checksum_validation="when_required"),
    ) as s3:
        for bucket, keys in s3_keys.items():
            for key in keys:
                try:
                    await s3.delete_object(Bucket=bucket, Key=key)
                except Exception:
                    logger.warning(
                        "delete_document.s3_failed",
                        extra={"bucket": bucket, "key": key},
                    )

        # Crops are one object per picture under `{document_id}/pictures/N.png`
        # (build_picture_crop_key), so this sweeps a prefix rather than deleting a fixed key.
        pictures_bucket = get_s3_pictures_bucket()
        prefix = f"{document_id}/"
        try:
            paginator = s3.get_paginator("list_objects_v2")
            async for page in paginator.paginate(Bucket=pictures_bucket, Prefix=prefix):
                batch = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
                if batch:
                    await s3.delete_objects(Bucket=pictures_bucket, Delete={"Objects": batch})
        except Exception:
            logger.warning(
                "delete_document.s3_failed",
                extra={"bucket": pictures_bucket, "key": prefix},
            )

    await session.execute(text("DELETE FROM chunks WHERE document_id = :id"), {"id": document_id})
    await session.execute(text("DELETE FROM documents WHERE id = :id"), {"id": document_id})
