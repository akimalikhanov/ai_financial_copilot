from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import JSON, BigInteger, ForeignKey, Integer, Text, text
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base


class Document(Base):
    """User-uploaded PDF document. Maps to documents table."""

    __tablename__ = "documents"

    id: Mapped[UUID] = mapped_column(primary_key=True, server_default="gen_random_uuid()")
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    conversation_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("conversations.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    original_filename: Mapped[str] = mapped_column(Text, nullable=False)
    storage_key: Mapped[str] = mapped_column(Text, nullable=False)
    content_type: Mapped[str] = mapped_column(
        Text, nullable=False, server_default="application/pdf"
    )
    file_size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="pending")
    processing_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    extracted_title: Mapped[str | None] = mapped_column(Text, nullable=True)
    ingest_time_seconds: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True,
        comment="JSON: {stages: {stage_name: seconds}, total_time: float}",
    )
    parse_status: Mapped[str | None] = mapped_column(Text, nullable=True)
    ingest_attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default="now()")
    updated_at: Mapped[datetime] = mapped_column(nullable=False, server_default="now()")
    document_metadata: Mapped[dict] = mapped_column(
        "metadata",
        JSON,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )
