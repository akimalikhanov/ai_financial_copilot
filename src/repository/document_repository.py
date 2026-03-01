from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.document import Document


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
