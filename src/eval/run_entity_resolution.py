"""Offline eval of entity resolution (exact match, candidates, LLM disambiguator) against a
seeded library, with real LLM calls.

Usage:
    python -m src.eval.run_entity_resolution [--user-id <uuid>] \
        [--cases src/eval/fixtures/entity_resolution_eval.json]

Each case names one entity. It passes when the resolved company's name contains one of
``expect``, or when ``expect`` is empty and nothing resolves. Cases whose expected company
isn't in the library are skipped.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from uuid import UUID

from src.db.connection import get_session_factory, init_db, shutdown_db
from src.repository.document_repository import DocumentRepository
from src.schemas.query_router import ExtractedEntity
from src.services.router.entity_resolver import resolve_entities
from src.utils.config import get_eval_user_id

_DEFAULT_CASES = Path(__file__).parent / "fixtures" / "entity_resolution_eval.json"


async def _run(user_id: UUID, cases: list[dict]) -> int:
    await init_db()
    failed = 0
    try:
        async with get_session_factory()() as session:
            companies = await DocumentRepository(session).list_companies(user_id)
            names = [c.display_name for c in companies]
            for case in cases:
                if case["expect"] and not any(e in n for e in case["expect"] for n in names):
                    print(f"SKIP  {case['raw_span']!r}: {case['expect']} not in library")
                    continue
                entity = ExtractedEntity(
                    name=case["name"], entity_type="company", raw_span=case["raw_span"]
                )
                [res] = await resolve_entities(session, user_id, case["query"], [entity])
                got = res.candidates[0].display_name if res.decision != "none" else None
                ok = any(e in got for e in case["expect"]) if got else not case["expect"]
                failed += not ok
                print(
                    f"{'PASS' if ok else 'FAIL'}  {case['raw_span']!r} -> {res.decision} "
                    f"via {res.method}: {[c.display_name for c in res.candidates] or 'none'}"
                )
                await session.rollback()
    finally:
        await shutdown_db()
    print(f"\n{len(cases)} cases, {failed} failed")
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline eval of entity resolution")
    parser.add_argument("--user-id", type=UUID, default=get_eval_user_id())
    parser.add_argument("--cases", type=Path, default=_DEFAULT_CASES)
    args = parser.parse_args()
    if args.user_id is None:
        parser.error("--user-id or EVAL_USER_ID is required")
    cases = json.loads(args.cases.read_text())
    raise SystemExit(1 if asyncio.run(_run(args.user_id, cases)) else 0)


if __name__ == "__main__":
    main()
