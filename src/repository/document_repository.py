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

    async def update_status(self, document_id: UUID, status: str) -> bool:
        """Update document status. Returns True if a row was updated."""
        from sqlalchemy import update

        from src.models.document import Document

        result = await self.session.execute(
            update(Document).where(Document.id == document_id).values(status=status)
        )
        await self.session.flush()
        return getattr(result, "rowcount", 0) > 0

    async def list_by_user(self, user_id: UUID) -> list[Document]:
        """List documents owned by a user (newest first)."""
        result = await self.session.execute(
            select(Document).where(Document.user_id == user_id).order_by(Document.created_at.desc())
        )
        return list(result.scalars().all())
