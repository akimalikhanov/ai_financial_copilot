from __future__ import annotations

import contextlib
from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.document import Document
from src.schemas.query_router import CompanyCandidate

# One candidate per company_norm, aggregated over its documents. Shared by the candidate
# query and the catalogue listing so both return the same shape.
_CANDIDATE_COLUMNS = """
    company_norm,
    min(metadata->>'company') AS display_name,
    array_agg(id) AS doc_ids,
    coalesce(array_agg(DISTINCT (metadata->>'year')::int)
             FILTER (WHERE metadata->>'year' ~ '^[0-9]+$'), '{}') AS years,
    coalesce(array_agg(DISTINCT extracted_title)
             FILTER (WHERE extracted_title IS NOT NULL), '{}') AS titles
"""


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
        id: UUID | None = None,
        conversation_id: UUID | None = None,
        content_type: str = "application/pdf",
        file_size_bytes: int | None = None,
        metadata: dict | None = None,
        company_norm: str | None = None,
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
            company_norm=company_norm,
        )
        if id is not None:
            doc.id = id
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

        # `where` interpolates only column names/bind-param placeholders built from
        # enumerate(); all filter values are passed via `params`, not string-formatted.
        where = " AND ".join(conditions)
        sql = f"SELECT id FROM documents WHERE {where}"
        stmt = text(sql)  # nosemgrep
        rows = (await self.session.execute(stmt, params)).fetchall()  # noqa: S608
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

    async def get_scope_docs(
        self, user_id: UUID, doc_ids: list[UUID] | None = None
    ) -> list[tuple[UUID, str | None, str | None, int | None]]:
        """(id, company_norm, company, year) for the user's ready docs among ``doc_ids``,
        or all of them when ``doc_ids`` is None."""
        rows = (
            await self.session.execute(
                text("""
                    SELECT
                        id,
                        company_norm,
                        metadata->>'company',
                        CASE WHEN metadata->>'year' ~ '^[0-9]+$'
                             THEN (metadata->>'year')::int
                        END
                    FROM documents
                    WHERE user_id = CAST(:user_id AS uuid)
                      AND status = 'ready'
                      AND (CAST(:doc_ids AS uuid[]) IS NULL
                           OR id = ANY(CAST(:doc_ids AS uuid[])))
                    ORDER BY created_at
                """),
                {
                    "user_id": str(user_id),
                    "doc_ids": [str(d) for d in doc_ids] if doc_ids is not None else None,
                },
            )
        ).fetchall()
        return [(UUID(str(r[0])), r[1], r[2] or None, r[3]) for r in rows]

    async def find_company_candidates(
        self,
        user_id: UUID,
        keys: list[str],
        *,
        limit: int,
        sim_threshold: float,
        word_threshold: float,
    ) -> dict[str, list[CompanyCandidate]]:
        """Companies matching each normalized lookup key: exact, `%` (similarity) or `<<%`
        (strict_word_similarity), best first, up to ``limit`` per key. Every key is in the
        result, unmatched ones with an empty list."""
        keys = list(dict.fromkeys(keys))
        result: dict[str, list[CompanyCandidate]] = {k: [] for k in keys}
        if not keys:
            return result
        # Transaction-local, so safe under pgbouncer transaction pooling.
        await self.session.execute(
            text(
                "SELECT set_config('pg_trgm.similarity_threshold', :sim, true),"
                " set_config('pg_trgm.strict_word_similarity_threshold', :word, true)"
            ),
            {"sim": str(sim_threshold), "word": str(word_threshold)},
        )
        rows = await self.session.execute(
            text(f"""
                SELECT k.key, c.*
                FROM unnest(CAST(:keys AS text[])) AS k(key)
                CROSS JOIN LATERAL (
                    SELECT {_CANDIDATE_COLUMNS},
                           greatest(max(similarity(company_norm, k.key)),
                                    max(strict_word_similarity(k.key, company_norm))) AS score
                    FROM documents
                    WHERE user_id = CAST(:user_id AS uuid)
                      AND status = 'ready'
                      AND (company_norm = k.key
                           OR k.key <<% company_norm
                           OR k.key % company_norm)
                    GROUP BY company_norm
                    ORDER BY score DESC
                    LIMIT :limit
                ) c
            """),  # nosemgrep
            {"user_id": str(user_id), "keys": keys, "limit": limit},
        )
        for row in rows.mappings():
            result[row["key"]].append(CompanyCandidate.model_validate(dict(row)))
        return result

    async def list_companies(
        self, user_id: UUID, *, limit: int | None = None
    ) -> list[CompanyCandidate]:
        """Every company among the user's ready documents, in the candidate shape, up to
        ``limit`` (None for all)."""
        rows = await self.session.execute(
            text(f"""
                SELECT {_CANDIDATE_COLUMNS}
                FROM documents
                WHERE user_id = CAST(:user_id AS uuid)
                  AND status = 'ready'
                  AND company_norm IS NOT NULL
                GROUP BY company_norm
                ORDER BY company_norm
                LIMIT :limit
            """),  # nosemgrep
            {"user_id": str(user_id), "limit": limit},
        )
        return [CompanyCandidate.model_validate(dict(row)) for row in rows.mappings()]
