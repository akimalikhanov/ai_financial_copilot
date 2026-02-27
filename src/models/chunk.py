from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import JSON, ForeignKey, Integer, Text, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base


class Chunk(Base):
    """Document chunk for retrieval/indexing. Maps to chunks table."""

    __tablename__ = "chunks"

    id: Mapped[UUID] = mapped_column(primary_key=True, server_default="gen_random_uuid()")
    document_id: Mapped[UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    enriched_text: Mapped[str] = mapped_column(Text, nullable=False)
    heading_trail: Mapped[list[str] | None] = mapped_column(ARRAY(Text()), nullable=True)
    chunk_type: Mapped[str] = mapped_column(Text, nullable=False)
    page_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    provenance: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        server_default=text("'[]'::jsonb"),
    )
    embedding_model: Mapped[str | None] = mapped_column(Text, nullable=True)
    chunk_metadata: Mapped[dict] = mapped_column(
        "metadata",
        JSON,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default="now()")

    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "chunk_index",
            name="chunks_document_id_chunk_index_key",
        ),
    )
