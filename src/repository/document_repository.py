from __future__ import annotations

import contextlib
import re
from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.document import Document

_CORP_SUFFIXES = (
    # English
    "limited",
    "ltd",
    "llc",
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "company",
    "plc",
    "lp",
    "llp",
    "pllc",
    "pc",
    "holdings",
    "holding",
    "group",
    "trust",
    "reit",
    # Australian/UK/SG/MY
    "pty",
    "pte",
    "sdn",
    "bhd",
    # German/Austrian/Swiss
    "gmbh",
    "ag",
    "kg",
    "kgaa",
    "se",
    # Dutch
    "bv",
    "nv",
    "cv",
    # French/Belgian
    "sa",
    "sas",
    "sarl",
    "sca",
    # Italian/Spanish/Portuguese
    "srl",
    "spa",
    "sl",
    # Nordic
    "ab",
    "oy",
    "oyj",
    "as",
    "asa",
    "aps",
    # Japanese romanized
    "kk",
    "gk",
)

# Generic industry/sector words that appear in many company names and reduce
# discriminative power when doing trigram similarity matching.
_INDUSTRY_WORDS = (
    "pharmaceuticals",
    "pharmaceutical",
    "pharma",
    "biosciences",
    "bioscience",
    "biotechnology",
    "biotech",
    "therapeutics",
    "therapies",
    "sciences",
    "science",
    "laboratories",
    "laboratory",
    "labs",
    "technologies",
    "technology",
    "solutions",
    "services",
    "systems",
    "industries",
    "international",
    "global",
    "national",
    "enterprises",
    "partners",
    "capital",
    "ventures",
    "financial",
    "investments",
    "management",
    "resources",
)

_CORP_SUFFIX_RE = re.compile(
    r"\b(" + "|".join(_CORP_SUFFIXES) + r")\b\.?\s*$",
    re.IGNORECASE,
)
_INDUSTRY_RE = re.compile(
    r"\b(" + "|".join(_INDUSTRY_WORDS) + r")\b",
    re.IGNORECASE,
)
# \m is PostgreSQL's start-of-word anchor; pattern interpolated into SQL literals.
_CORP_SUFFIX_SQL_RE = r"\m(" + "|".join(_CORP_SUFFIXES) + r")\.?\s*$"
_INDUSTRY_SQL_RE = r"\m(" + "|".join(_INDUSTRY_WORDS) + r")\M"
_MIN_NORMALIZED_LEN = 3


def _normalize_company(name: str) -> str:
    """Strip trailing corp suffixes, punctuation, and generic industry words.

    Falls back to the original (lowercased) string when stripping would leave
    fewer than _MIN_NORMALIZED_LEN characters (e.g. "The Limited" → "the").
    """
    lowered = name.lower().strip()
    # Strip trailing corp suffix first
    stripped = _CORP_SUFFIX_RE.sub("", lowered).strip()
    # Strip trailing punctuation (commas, periods left behind by "Ltd., Inc." etc.)
    stripped = stripped.rstrip(".,;").strip()
    # Remove generic industry words that inflate cross-company similarity
    stripped = _INDUSTRY_RE.sub("", stripped).strip()
    # Collapse multiple spaces
    stripped = re.sub(r"\s+", " ", stripped).strip()
    return stripped if len(stripped) >= _MIN_NORMALIZED_LEN else lowered


class DocumentRepository:
    """Repository for document CRUD operations."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        user_id: UUID,
        original_filename: str,
        storage_key: str,
        *,
        conversation_id: UUID | None = None,
        content_type: str = "application/pdf",
        file_size_bytes: int | None = None,
        metadata: dict | None = None,
    ) -> Document:
        """Create a new document record."""
        doc = Document(
            user_id=user_id,
            original_filename=original_filename,
            storage_key=storage_key,
            conversation_id=conversation_id,
            content_type=content_type,
            file_size_bytes=file_size_bytes,
            document_metadata=metadata or {},
        )
        self.session.add(doc)
        await self.session.flush()
        return doc

    async def update_status(
        self, document_id: UUID, status: str, *, clear_processing_error: bool = False
    ) -> bool:
        """Update document status. Returns True if a row was updated."""
        from sqlalchemy import update

        from src.models.document import Document

        values: dict[str, str | None] = {"status": status}
        if clear_processing_error:
            values["processing_error"] = None

        result = await self.session.execute(
            update(Document).where(Document.id == document_id).values(**values)
        )
        await self.session.flush()
        return getattr(result, "rowcount", 0) > 0

    async def get_by_id(self, document_id: UUID) -> Document | None:
        return await self.session.get(Document, document_id)

    async def update_metadata(
        self,
        document_id: UUID,
        *,
        page_count: int | None = None,
        extracted_title: str | None = None,
        parse_status: str | None = None,
        metadata: dict | None = None,
    ) -> bool:
        from sqlalchemy import update

        values: dict = {}
        if page_count is not None:
            values["page_count"] = page_count
        if extracted_title is not None:
            values["extracted_title"] = extracted_title
        if parse_status is not None:
            values["parse_status"] = parse_status
        if metadata is not None:
            values["document_metadata"] = metadata
        if not values:
            return False

        result = await self.session.execute(
            update(Document).where(Document.id == document_id).values(**values)
        )
        await self.session.flush()
        return getattr(result, "rowcount", 0) > 0

    async def set_failed(self, document_id: UUID, error: str) -> bool:
        from sqlalchemy import update

        result = await self.session.execute(
            update(Document)
            .where(Document.id == document_id)
            .values(status="failed", processing_error=error)
        )
        await self.session.flush()
        return getattr(result, "rowcount", 0) > 0

    async def increment_attempt_count(self, document_id: UUID) -> int:
        """Atomically increment and return the new attempt count."""
        from sqlalchemy import update

        result = await self.session.execute(
            update(Document)
            .where(Document.id == document_id)
            .values(ingest_attempt_count=Document.ingest_attempt_count + 1)
            .returning(Document.ingest_attempt_count)
        )
        await self.session.flush()
        return result.scalar_one()

    async def set_ingest_time_seconds(
        self, document_id: UUID, ingest_times: dict[str, object]
    ) -> bool:
        from sqlalchemy import update

        result = await self.session.execute(
            update(Document)
            .where(Document.id == document_id)
            .values(ingest_time_seconds=ingest_times)
        )
        await self.session.flush()
        return getattr(result, "rowcount", 0) > 0

    async def list_by_user(self, user_id: UUID) -> list[Document]:
        """List documents owned by a user (newest first)."""
        result = await self.session.execute(
            select(Document).where(Document.user_id == user_id).order_by(Document.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_by_ids(self, doc_ids: list[UUID]) -> list[Document]:
        if not doc_ids:
            return []
        result = await self.session.execute(select(Document).where(Document.id.in_(doc_ids)))
        return list(result.scalars().all())

    async def list_ready_by_user(self, user_id: UUID) -> list[Document]:
        """List documents owned by a user with status 'ready'."""
        result = await self.session.execute(
            select(Document).where(
                Document.user_id == user_id,
                Document.status == "ready",
            )
        )
        return list(result.scalars().all())

    async def find_by_metadata_filters(
        self,
        user_id: UUID,
        *,
        companies: list[str] | None = None,
        years: list[int] | None = None,
        types: list[str] | None = None,
    ) -> list[UUID]:
        """Find ready document IDs matching metadata filters (ILIKE for strings, = for year)."""
        conditions = ["user_id = CAST(:user_id AS uuid)", "status = 'ready'"]
        params: dict = {"user_id": str(user_id)}

        if companies:
            clauses = []
            for i, c in enumerate(companies):
                key = f"company_{i}"
                clauses.append(f"metadata->>'company' ILIKE :{key}")
                params[key] = f"%{c}%"
            conditions.append(f"({' OR '.join(clauses)})")

        if years:
            params["years"] = years
            conditions.append("(metadata->>'year')::int = ANY(:years)")

        if types:
            clauses = []
            for i, t in enumerate(types):
                key = f"type_{i}"
                clauses.append(f"metadata->>'type' ILIKE :{key}")
                params[key] = f"%{t}%"
            conditions.append(f"({' OR '.join(clauses)})")

        where = " AND ".join(conditions)
        rows = (
            await self.session.execute(text(f"SELECT id FROM documents WHERE {where}"), params)
        ).fetchall()  # noqa: S608
        return [row[0] for row in rows]

    async def get_filter_options(self, user_id: UUID) -> dict[str, list]:
        """Return distinct non-null companies and years for a user's ready documents."""
        rows = (
            await self.session.execute(
                text(
                    """
                    SELECT
                        metadata->>'company' AS company,
                        metadata->>'year'    AS year
                    FROM documents
                    WHERE user_id = CAST(:user_id AS uuid)
                      AND status = 'ready'
                      AND (metadata->>'company' IS NOT NULL OR metadata->>'year' IS NOT NULL)
                    """
                ),
                {"user_id": str(user_id)},
            )
        ).fetchall()

        companies: set[str] = set()
        years: set[int] = set()
        for company, year in rows:
            if company:
                companies.add(company)
            if year:
                with contextlib.suppress(ValueError):
                    years.add(int(year))

        return {
            "companies": sorted(companies),
            "years": sorted(years, reverse=True),
        }

    async def get_scope_doc_summaries(
        self, user_id: UUID, doc_ids: list[UUID], *, limit: int
    ) -> list[tuple[UUID, str | None, int | None]]:
        """Return (id, company, year) for ready docs in scope."""
        if not doc_ids:
            return []
        rows = (
            await self.session.execute(
                text("""
                    SELECT
                        id,
                        metadata->>'company',
                        CASE WHEN metadata->>'year' ~ '^[0-9]+$'
                             THEN (metadata->>'year')::int
                        END
                    FROM documents
                    WHERE user_id = CAST(:user_id AS uuid)
                      AND id = ANY(CAST(:doc_ids AS uuid[]))
                      AND status = 'ready'
                    LIMIT :limit
                """),
                {"user_id": str(user_id), "doc_ids": [str(d) for d in doc_ids], "limit": limit},
            )
        ).fetchall()
        return [
            (UUID(str(row[0])), row[1] or None, int(row[2]) if row[2] is not None else None)
            for row in rows
        ]

    async def find_by_company_similarity(
        self,
        user_id: UUID,
        name: str,
        *,
        threshold: float = 0.5,
        constrain_to: list[UUID] | None = None,
        limit: int = 10,
    ) -> list[UUID]:
        """Find document IDs by fuzzy-matching metadata company name (pg_trgm).

        Returns doc IDs ordered by similarity descending, up to ``limit``.
        """
        params: dict = {
            "user_id": str(user_id),
            "name": _normalize_company(name),
            "threshold": threshold,
            "limit": limit,
        }

        constrain_clause = ""
        if constrain_to:
            params["constrain_to"] = [str(d) for d in constrain_to]
            constrain_clause = "AND id = ANY(CAST(:constrain_to AS uuid[]))"

        stmt = text(f"""
            SELECT id FROM documents
            WHERE user_id = CAST(:user_id AS uuid)
              AND status = 'ready'
              AND metadata->>'company' IS NOT NULL
              AND similarity(
                    trim(regexp_replace(
                        regexp_replace(
                            lower(metadata->>'company'),
                            '{_CORP_SUFFIX_SQL_RE}',
                            '', 'i'
                        ),
                        '{_INDUSTRY_SQL_RE}',
                        '', 'gi'
                    )),
                    :name
                  ) > :threshold
              {constrain_clause}
            ORDER BY similarity(
                trim(regexp_replace(
                    regexp_replace(
                        lower(metadata->>'company'),
                        '{_CORP_SUFFIX_SQL_RE}',
                        '', 'i'
                    ),
                    '{_INDUSTRY_SQL_RE}',
                    '', 'gi'
                )),
                :name
            ) DESC
            LIMIT :limit
        """)

        rows = (await self.session.execute(stmt, params)).fetchall()
        return [row[0] for row in rows]
