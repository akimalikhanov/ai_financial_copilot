"""Find and delete S3 objects with no matching document row in the DB.

Two different checks are used depending on the bucket:

  pdfs bucket  — keys are  raw/{user_id}/{upload_uuid}/{filename}
                 The UUID at position [2] was generated before the DB row existed
                 (old upload code), so it does NOT equal document.id.
                 Orphan check: compare the full key against documents.storage_key.

  docling / rendered / chunks buckets — keys are
                 processed/{user_id}/{document_id}/{artifact}
                 The UUID at position [2] IS documents.id (written by the
                 ingestion worker after the row is committed).
                 Orphan check: extract parts[2] and compare against document.id.

Dry-run by default.

Usage:
    .venv/bin/python -m scripts.cleanup_orphaned_s3            # dry-run
    .venv/bin/python -m scripts.cleanup_orphaned_s3 --yes      # actually delete
"""

from __future__ import annotations

import argparse
import asyncio
import re
from uuid import UUID

import aioboto3
from dotenv import load_dotenv

load_dotenv()

_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
)


def _s3_client():
    from src.utils.config import get_s3_access_key, get_s3_endpoint_url, get_s3_secret_key

    session = aioboto3.Session()
    return session.client(
        "s3",
        endpoint_url=get_s3_endpoint_url(),
        region_name="garage",
        aws_access_key_id=get_s3_access_key(),
        aws_secret_access_key=get_s3_secret_key(),
    )


async def _list_keys(bucket: str) -> list[str]:
    keys: list[str] = []
    async with _s3_client() as client:  # type: ignore[attr-defined]
        paginator = client.get_paginator("list_objects_v2")
        async for page in paginator.paginate(Bucket=bucket):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
    return keys


def _doc_id_from_processed_key(key: str) -> UUID | None:
    """Extract document.id from processed/{user_id}/{document_id}/{artifact} keys."""
    parts = key.split("/")
    if len(parts) < 3:
        return None
    candidate = parts[2]
    if _UUID_RE.fullmatch(candidate):
        return UUID(candidate)
    return None


async def main() -> None:
    parser = argparse.ArgumentParser(description="Delete orphaned S3 objects.")
    parser.add_argument("--yes", action="store_true", help="Actually delete (default: dry-run)")
    args = parser.parse_args()

    from sqlalchemy import select

    from src.db.connection import get_session_factory, init_db
    from src.models.document import Document
    from src.utils.config import (
        get_s3_chunks_bucket,
        get_s3_docling_bucket,
        get_s3_raw_bucket,
        get_s3_rendered_bucket,
    )

    await init_db()
    factory = get_session_factory()
    async with factory() as session:
        rows = (await session.execute(select(Document.id, Document.storage_key))).all()

    db_doc_ids: set[UUID] = {r.id for r in rows}
    # Full storage_key values recorded in the DB — used to identify live raw PDFs.
    # The UUID inside these keys was generated before the DB row existed, so it
    # does NOT match document.id and must be compared as a full key string.
    db_storage_keys: set[str] = {r.storage_key for r in rows}

    print(f"Documents in DB: {len(db_doc_ids)}")

    raw_bucket = get_s3_raw_bucket()
    processed_buckets = dict.fromkeys(
        [
            get_s3_docling_bucket(),
            get_s3_rendered_bucket(),
            get_s3_chunks_bucket(),
        ]
    )

    orphaned_by_bucket: dict[str, list[str]] = {}
    total_keys = 0

    # --- raw PDFs: compare full key against documents.storage_key ---
    raw_keys = await _list_keys(raw_bucket)
    total_keys += len(raw_keys)
    orphaned_by_bucket[raw_bucket] = [k for k in raw_keys if k not in db_storage_keys]

    # --- processed artifacts: compare parts[2] (= document.id) against document.id ---
    for bucket in processed_buckets:
        keys = await _list_keys(bucket)
        total_keys += len(keys)
        orphaned = []
        for key in keys:
            doc_id = _doc_id_from_processed_key(key)
            if doc_id is None:
                print(f"  [skip] s3://{bucket}/{key}: could not parse document_id, leaving alone")
                continue
            if doc_id not in db_doc_ids:
                orphaned.append(key)
        orphaned_by_bucket[bucket] = orphaned

    print(f"Total objects scanned: {total_keys}")
    for bucket, keys in orphaned_by_bucket.items():
        print(f"\n{bucket}: {len(keys)} orphaned object(s)")
        for k in keys:
            print(f"  s3://{bucket}/{k}")

    grand_total = sum(len(k) for k in orphaned_by_bucket.values())
    if grand_total == 0:
        print("\nNo orphans found.")
        return

    if not args.yes:
        print(f"\n[dry-run] {grand_total} object(s) would be deleted. Re-run with --yes to delete.")
        return

    print(f"\nDeleting {grand_total} object(s)...")
    async with _s3_client() as client:  # type: ignore[attr-defined]
        for bucket, keys in orphaned_by_bucket.items():
            for key in keys:
                try:
                    await client.delete_object(Bucket=bucket, Key=key)
                    print(f"  deleted s3://{bucket}/{key}")
                except Exception as e:
                    print(f"  [warn] s3://{bucket}/{key}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
