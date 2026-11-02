"""Fill documents.company_norm from metadata.company with normalize_company.

Idempotent: only rows whose stored value differs are written, so rerun it after any
change to the normalizer.

Usage:
    .venv/bin/python -m scripts.backfill_company_norm [--dry-run] [--batch-size N]
"""

from __future__ import annotations

import argparse
import asyncio

from dotenv import load_dotenv

load_dotenv()


async def backfill(*, dry_run: bool, batch_size: int) -> None:
    from sqlalchemy import text

    from src.db.connection import get_session_factory, init_db
    from src.services.router.company_name import normalize_company

    await init_db()
    factory = get_session_factory()
    last_id = None
    changed = 0
    while True:
        async with factory() as session:
            rows = (
                await session.execute(
                    text(
                        """
                        SELECT id, metadata->>'company', company_norm FROM documents
                        WHERE metadata->>'company' IS NOT NULL
                          AND (CAST(:last_id AS uuid) IS NULL OR id > CAST(:last_id AS uuid))
                        ORDER BY id
                        LIMIT :limit
                        """
                    ),
                    {"last_id": last_id, "limit": batch_size},
                )
            ).fetchall()
            if not rows:
                break
            last_id = str(rows[-1][0])
            updates = []
            for doc_id, company, old in rows:
                new = normalize_company(company)
                if new != old:
                    print(f"{doc_id}  {company!r}: {old!r} -> {new!r}")
                    updates.append({"id": doc_id, "norm": new})
            changed += len(updates)
            if updates and not dry_run:
                await session.execute(
                    text("UPDATE documents SET company_norm = :norm WHERE id = :id"), updates
                )
                await session.commit()
    print(f"{'Would update' if dry_run else 'Updated'} {changed} document(s).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill documents.company_norm.")
    parser.add_argument("--dry-run", action="store_true", help="Print changes, write nothing")
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()
    asyncio.run(backfill(dry_run=args.dry_run, batch_size=args.batch_size))


if __name__ == "__main__":
    main()
