from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import JSON, ForeignKey, Integer, Text, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.base import Base

if TYPE_CHECKING:
    from src.models.message import Message


class Conversation(Base):
    """
    Conversation model representing a chat conversation.

    Maps to the conversations table in PostgreSQL.
    """

    __tablename__ = "conversations"

    # Primary key
    id: Mapped[UUID] = mapped_column(primary_key=True, server_default="gen_random_uuid()")

    # Foreign keys
    user_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,  # Made nullable for skip-auth approach
        index=True,
    )

    # Conversation metadata
    title: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Denormalized fields for performance
    last_message_at: Mapped[datetime | None] = mapped_column(nullable=True)
    last_message_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("messages.id", ondelete="SET NULL"),
        nullable=True,
    )
    message_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default="0",
    )

    # UI state
    pinned: Mapped[bool] = mapped_column(
        nullable=False,
        server_default="false",
    )
    archived_at: Mapped[datetime | None] = mapped_column(nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(nullable=True, index=True)

    # Summary for long conversations
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary_updated_at: Mapped[datetime | None] = mapped_column(nullable=True)

    # Configuration and metadata
    settings: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )
    metadata: Mapped[dict] = mapped_column(  # type: ignore[assignment]
        JSON,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        nullable=False,
        server_default="now()",
    )
    updated_at: Mapped[datetime] = mapped_column(
        nullable=False,
        server_default="now()",
    )

    # Relationships
    messages: Mapped[list[Message]] = relationship(
        "Message",
        back_populates="conversation",
        order_by="Message.seq",
        cascade="all, delete-orphan",
    )
