"""Wipe a document completely from all storage layers: DB, Qdrant, OpenSearch, S3.

Usage:
    .venv/bin/python -m scripts.wipe_document <document_id> [--yes]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from uuid import UUID

import aioboto3
from dotenv import load_dotenv

load_dotenv()


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


async def _delete_s3_keys(bucket: str, keys: list[str]) -> list[str]:
    deleted = []
    async with _s3_client() as client:  # type: ignore[attr-defined]
        for key in keys:
            try:
                await client.delete_object(Bucket=bucket, Key=key)
                deleted.append(f"s3://{bucket}/{key}")
            except Exception as e:
                print(f"  [warn] s3://{bucket}/{key}: {e}")
    return deleted


async def wipe_document(doc_id: UUID, *, dry_run: bool = False) -> None:
    from sqlalchemy import select, text

    from src.db.connection import get_session_factory, init_db
    from src.models.chunk import Chunk
    from src.models.document import Document
    from src.services.ingestion import opensearch_ingest, qdrant_ingest
    from src.utils.config import (
        get_s3_chunks_bucket,
        get_s3_docling_bucket,
        get_s3_raw_bucket,
        get_s3_rendered_bucket,
    )

    qdrant_collection = os.getenv("QDRANT_COLLECTION", "documents")
    opensearch_index = os.getenv("OPENSEARCH_INDEX", "chunks")

    await init_db()
    factory = get_session_factory()

    async with factory() as session:
        doc = await session.get(Document, doc_id)
        if doc is None:
            print(f"Document {doc_id} not found in DB.")
            sys.exit(1)

        chunks = (
            (await session.execute(select(Chunk).where(Chunk.document_id == doc_id)))
            .scalars()
            .all()
        )

        user_id = doc.user_id
        storage_key = doc.storage_key
        base_key = f"processed/{user_id}/{doc_id}"

        s3_keys = {
            get_s3_raw_bucket(): [storage_key],
            get_s3_docling_bucket(): [f"{base_key}/docling.json"],
            get_s3_rendered_bucket(): [f"{base_key}/document.md"],
            get_s3_chunks_bucket(): [f"{base_key}/chunks.jsonl"],
        }

    print(f"\nDocument  : {doc_id}")
    print(f"Filename  : {doc.original_filename}")
    print(f"User      : {user_id}")
    print(f"Status    : {doc.status}")
    print(f"Chunks    : {len(chunks)}")
    print("\nS3 objects to delete:")
    for bucket, keys in s3_keys.items():
        for k in keys:
            print(f"  s3://{bucket}/{k}")
    print(f"\nQdrant    : collection={qdrant_collection!r}, filter document_id={doc_id}")
    print(f"OpenSearch: index={opensearch_index!r}, filter document_id={doc_id}")
    print(f"DB        : {len(chunks)} chunks + 1 document row (CASCADE)")

    if dry_run:
        print("\n[dry-run] No changes made.")
        return

    print()

    # 1. Qdrant
    print("Deleting Qdrant vectors...")
    try:
        qdrant_ingest.delete_by_document(qdrant_collection, doc_id)
        print("  OK")
    except Exception as e:
        print(f"  [warn] {e}")

    # 2. OpenSearch
    print("Deleting OpenSearch docs...")
    try:
        opensearch_ingest.delete_by_document(opensearch_index, doc_id)
        print("  OK")
    except Exception as e:
        print(f"  [warn] {e}")

    # 3. S3
    print("Deleting S3 objects...")
    for bucket, keys in s3_keys.items():
        deleted = await _delete_s3_keys(bucket, keys)
        for d in deleted:
            print(f"  deleted {d}")

    # 4. DB (chunks cascade via FK, then document)
    print("Deleting from DB...")
    async with factory() as session:
        await session.execute(text("DELETE FROM chunks WHERE document_id = :id"), {"id": doc_id})
        await session.execute(text("DELETE FROM documents WHERE id = :id"), {"id": doc_id})
        await session.commit()
    print("  OK")

    print(f"\nDocument {doc_id} fully wiped.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Wipe a document from all storage layers.")
    parser.add_argument("document_id", type=UUID, help="UUID of the document to delete")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print what would be deleted without doing it"
    )
    args = parser.parse_args()

    if not args.yes and not args.dry_run:
        confirm = input(
            f"\nThis will permanently delete document {args.document_id} from ALL systems. Type 'yes' to confirm: "
        )
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            sys.exit(0)

    asyncio.run(wipe_document(args.document_id, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
