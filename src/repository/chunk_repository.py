from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.chunk import Chunk


class ChunkRepository:
    """Repository for chunk CRUD operations."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_many(self, document_id: UUID, chunks: Sequence[dict]) -> list[Chunk]:
        objs: list[Chunk] = []
        for c in chunks:
            objs.append(
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
            )

        self.session.add_all(objs)
        await self.session.flush()
        return objs

    async def list_by_document(self, document_id: UUID) -> list[Chunk]:
        result = await self.session.execute(
            select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.chunk_index.asc())
        )
        return list(result.scalars().all())
