"""Integration test: company candidate generation against real pg_trgm.

Requires: PostgreSQL running (docker-compose up -d postgres pgbouncer), with
`documents.company_norm`. Everything runs in one transaction that is rolled back.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from uuid import UUID, uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from src.models.document import Document
from src.models.user import User
from src.repository.document_repository import DocumentRepository
from src.services.router.company_name import normalize_company
from src.utils.config import get_db_url

COMPANIES = [
    "Aurora Innovation, Inc.",
    "Aurora Mobile Limited",
    "Poste Italiane S.p.A.",
    "Microsoft Corporation (scanned 10p)",
    "AA Limited",
    "Capital One Financial",
    "Elixir Energy Limited",
    "RWE AG",
]


@pytest.fixture
async def session() -> AsyncGenerator[AsyncSession, None]:
    engine = create_async_engine(get_db_url(), poolclass=NullPool)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as s:
        yield s
        await s.rollback()
    await engine.dispose()


@pytest.fixture
async def user_id(session: AsyncSession) -> UUID:
    user = User(email=f"candidates-{uuid4().hex[:8]}@test.com", password_hash="unused")
    session.add(user)
    await session.flush()
    for company in COMPANIES:
        session.add(
            Document(
                user_id=user.id,
                original_filename=f"{company}.pdf",
                storage_key=f"uploads/{uuid4()}.pdf",
                status="ready",
                document_metadata={"company": company, "year": "2023"},
                company_norm=normalize_company(company),
            )
        )
    await session.flush()
    return user.id


async def _candidates(session: AsyncSession, user_id: UUID, name: str) -> list[str]:
    key = normalize_company(name)
    found = await DocumentRepository(session).find_company_candidates(
        user_id, [key], limit=20, sim_threshold=0.2, word_threshold=0.25
    )
    return [c.display_name for c in found[key]]


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("aurora", "Aurora Innovation, Inc."),
        ("auorra", "Aurora Innovation, Inc."),
        ("poste", "Poste Italiane S.p.A."),
        ("microsoft", "Microsoft Corporation (scanned 10p)"),
        ("Microsoft Corporation", "Microsoft Corporation (scanned 10p)"),
        ("the aa", "AA Limited"),
        ("Capital One Financial", "Capital One Financial"),
    ],
)
async def test_name_finds_its_company(
    session: AsyncSession, user_id: UUID, name: str, expected: str
) -> None:
    assert expected in await _candidates(session, user_id, name)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_two_auroras_are_two_candidates(session: AsyncSession, user_id: UUID) -> None:
    found = await _candidates(session, user_id, "aurora")
    assert {"Aurora Innovation, Inc.", "Aurora Mobile Limited"} <= set(found)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_unknown_name_has_no_candidates(session: AsyncSession, user_id: UUID) -> None:
    assert await _candidates(session, user_id, "Tesla") == []


@pytest.mark.integration
@pytest.mark.asyncio
async def test_candidate_carries_docs_and_years(session: AsyncSession, user_id: UUID) -> None:
    [rwe] = (
        await DocumentRepository(session).find_company_candidates(
            user_id, ["rwe"], limit=20, sim_threshold=0.2, word_threshold=0.25
        )
    )["rwe"]
    assert rwe.company_norm == "rwe"
    assert len(rwe.doc_ids) == 1 and rwe.years == [2023] and rwe.score == 1.0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_list_companies_returns_every_company(session: AsyncSession, user_id: UUID) -> None:
    companies = await DocumentRepository(session).list_companies(user_id)
    assert sorted(c.display_name for c in companies) == sorted(COMPANIES)
