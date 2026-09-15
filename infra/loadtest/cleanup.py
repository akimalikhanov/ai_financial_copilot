"""Delete the residue a load test leaves behind (docs/notes/loadtest-readiness-audit.md §8).

Every run auto-registers a throwaway user per simulated client (locustfile.py's
`loadtest-{uuid4}@example.com`), and nothing ever collects them. After the chat track's T1-T4
plus two ingestion baselines this stood at 882 users, 878 conversations, 11,462 messages and
10 documents — enough to skew any Grafana or eval query that counts users or messages, and it
only grows.

Runs INSIDE the cluster (see the `k8s-loadtest-cleanup` make target): it needs the same Qdrant,
OpenSearch and S3 endpoints the app uses, and the API's own DELETE /v1/documents/{id} is not
usable here because it authenticates as the owning user and these accounts have random
passwords nobody kept.

Order is load-bearing, and it is the same order delete_document uses: the search indexes and
S3 objects are keyed by document id, so they must go BEFORE the Postgres rows. Delete the rows
first and the surviving index entries are unreachable orphans with nothing left to name them.

Postgres itself needs one statement — every relevant FK cascades from users (conversations,
messages, sessions, llm_requests, documents, chunks, message_feedback).

Usage:
    python -m infra.loadtest.cleanup --dry-run
    python -m infra.loadtest.cleanup --yes
"""

from __future__ import annotations

import argparse
import asyncio
import os
from uuid import UUID

from botocore.config import Config
from sqlalchemy import text

# Only users matching this are ever touched. Anchored on both ends so a real address that
# merely contains the word cannot match.
LOADTEST_EMAIL_LIKE = "loadtest-%@example.com"

_COUNT_SQL = """
WITH lt AS (SELECT id FROM users WHERE email LIKE :pat)
SELECT 'users' AS t, count(*) AS n FROM lt
UNION ALL SELECT 'documents', count(*) FROM documents WHERE user_id IN (SELECT id FROM lt)
UNION ALL SELECT 'chunks', count(*) FROM chunks
    WHERE document_id IN (SELECT id FROM documents WHERE user_id IN (SELECT id FROM lt))
UNION ALL SELECT 'conversations', count(*) FROM conversations WHERE user_id IN (SELECT id FROM lt)
UNION ALL SELECT 'messages', count(*) FROM messages
    WHERE conversation_id IN (SELECT id FROM conversations WHERE user_id IN (SELECT id FROM lt))
UNION ALL SELECT 'sessions', count(*) FROM sessions WHERE user_id IN (SELECT id FROM lt)
"""

_DOCS_SQL = """
SELECT d.id, d.user_id, d.storage_key
FROM documents d
JOIN users u ON u.id = d.user_id
WHERE u.email LIKE :pat
"""


async def _purge_s3(docs: list[tuple[UUID, UUID, str]]) -> int:
    import aioboto3

    from src.utils.config import (
        get_s3_access_key,
        get_s3_chunks_bucket,
        get_s3_docling_bucket,
        get_s3_endpoint_url,
        get_s3_raw_bucket,
        get_s3_rendered_bucket,
        get_s3_secret_key,
    )

    deleted = 0
    session = aioboto3.Session()
    async with session.client(  # type: ignore[attr-defined]
        "s3",
        endpoint_url=get_s3_endpoint_url(),
        region_name="garage",
        aws_access_key_id=get_s3_access_key(),
        aws_secret_access_key=get_s3_secret_key(),
        config=Config(response_checksum_validation="when_required"),
    ) as s3:
        for doc_id, user_id, storage_key in docs:
            prefix = f"processed/{user_id}/{doc_id}"
            for bucket, key in (
                (get_s3_raw_bucket(), storage_key),
                (get_s3_docling_bucket(), f"{prefix}/docling.json"),
                (get_s3_rendered_bucket(), f"{prefix}/document.md"),
                (get_s3_chunks_bucket(), f"{prefix}/chunks.jsonl"),
            ):
                try:
                    await s3.delete_object(Bucket=bucket, Key=key)
                    deleted += 1
                except Exception as exc:  # noqa: BLE001 — best effort, same as the endpoint
                    print(f"    s3 {bucket}/{key}: {exc}")
    return deleted


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="count only, delete nothing")
    parser.add_argument("--yes", action="store_true", help="required to actually delete")
    args = parser.parse_args()

    from src.db.connection import get_session_factory, init_db, shutdown_db
    from src.services.ingestion import opensearch_ingest, qdrant_ingest

    await init_db()
    try:
        await _run(args, get_session_factory(), opensearch_ingest, qdrant_ingest)
    finally:
        await shutdown_db()


async def _run(args, session_factory, opensearch_ingest, qdrant_ingest) -> None:  # noqa: ANN001
    async with session_factory() as session:
        counts = (await session.execute(text(_COUNT_SQL), {"pat": LOADTEST_EMAIL_LIKE})).all()
        print(f"\nmatching {LOADTEST_EMAIL_LIKE!r}:")
        for table, n in counts:
            print(f"  {table:14} {n:6d}")

        docs = [
            (r[0], r[1], r[2])
            for r in (await session.execute(text(_DOCS_SQL), {"pat": LOADTEST_EMAIL_LIKE})).all()
        ]

        if args.dry_run or not args.yes:
            print("\ndry run — nothing deleted. Pass --yes to delete.")
            return

        collection = os.getenv("QDRANT_COLLECTION", "documents")
        index = os.getenv("OPENSEARCH_INDEX", "chunks")

        # Indexes and S3 first: they are keyed by document id, which only exists until the
        # Postgres rows go. A failure here must abort before the cascade, so it stays fixable.
        print(f"\nclearing search indexes for {len(docs)} document(s)...")
        for doc_id, _, _ in docs:
            await asyncio.gather(
                asyncio.to_thread(qdrant_ingest.delete_by_document, collection, doc_id),
                asyncio.to_thread(opensearch_ingest.delete_by_document, index, doc_id),
            )

        print("deleting S3 objects...")
        n_s3 = await _purge_s3(docs)
        print(f"  {n_s3} object(s) deleted")

        # One statement: every FK above cascades from users.
        print("deleting Postgres rows (cascading from users)...")
        result = await session.execute(
            text("DELETE FROM users WHERE email LIKE :pat"), {"pat": LOADTEST_EMAIL_LIKE}
        )
        await session.commit()
        print(f"  {result.rowcount} user(s) deleted, cascade applied")

    print("\ndone.")


if __name__ == "__main__":
    asyncio.run(main())
