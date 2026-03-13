from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.chunk import Chunk


class ChunkRepository:
    """Repository for chunk CRUD operations."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def delete_by_document(self, document_id: UUID) -> int:
        """Delete all chunks for a document. Returns number of rows deleted."""
        result = await self.session.execute(delete(Chunk).where(Chunk.document_id == document_id))
        await self.session.flush()
        return getattr(result, "rowcount", 0) or 0

    async def create_many(self, document_id: UUID, chunks: Sequence[dict]) -> list[Chunk]:
        """Delete existing chunks then insert new ones (full replace).

        Full DELETE + INSERT avoids orphan rows when chunk count changes
        between ingestion attempts.
        """
        if not chunks:
            return []

        await self.delete_by_document(document_id)

        objs = [
            Chunk(
                document_id=document_id,
                chunk_index=int(c["chunk_index"]),
                raw_text=str(c["raw_text"]),
                enriched_text=str(c["enriched_text"]),
                heading_trail=c.get("heading_trail"),
                chunk_type=str(c["chunk_type"]),
                page_start=c.get("page_start"),
                page_end=c.get("page_end"),
                token_count=c.get("token_count"),
                provenance=c.get("provenance") if c.get("provenance") is not None else [],
                embedding_model=c.get("embedding_model"),
                chunk_metadata=c.get("metadata") or {},
            )
            for c in chunks
        ]

        self.session.add_all(objs)
        await self.session.flush()
        return objs

    async def list_by_document(self, document_id: UUID) -> list[Chunk]:
        result = await self.session.execute(
            select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.chunk_index.asc())
        )
        return list(result.scalars().all())

    async def get_by_ids(self, chunk_ids: list[UUID]) -> list[Chunk]:
        if not chunk_ids:
            return []
        result = await self.session.execute(select(Chunk).where(Chunk.id.in_(chunk_ids)))
        return list(result.scalars().all())
