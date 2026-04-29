from __future__ import annotations

import enum
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import (
    JSON,
    BigInteger,
    ForeignKey,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy import (
    Enum as SQLEnum,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.base import Base

if TYPE_CHECKING:
    from src.models.conversation import Conversation
    from src.models.llm_request import LLMRequest


class MessageRole(str, enum.Enum):
    """Message role enum matching PostgreSQL message_role type."""

    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


class MessageStatus(str, enum.Enum):
    """Message status enum matching PostgreSQL message_status type."""

    completed = "completed"
    in_progress = "in_progress"
    cancelled = "cancelled"
    error = "error"


class Message(Base):
    """
    Message model representing a single message in a conversation.

    Maps to the messages table in PostgreSQL.
    """

    __tablename__ = "messages"

    # Primary key
    id: Mapped[UUID] = mapped_column(primary_key=True, server_default="gen_random_uuid()")

    # Foreign keys
    conversation_id: Mapped[UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    # Message content
    role: Mapped[MessageRole] = mapped_column(
        SQLEnum(MessageRole, native_enum=True, name="message_role"),
        nullable=False,
    )
    status: Mapped[MessageStatus] = mapped_column(
        SQLEnum(MessageStatus, native_enum=True, name="message_status"),
        nullable=False,
        server_default="completed",
    )
    seq: Mapped[int] = mapped_column(BigInteger, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    raw_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_format: Mapped[str] = mapped_column(
        String,
        nullable=False,
        server_default="text/markdown",
    )

    # Metadata
    message_metadata: Mapped[dict] = mapped_column(
        JSON,
        name="metadata",
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )

    # Pipeline trace (assistant messages only)
    trace: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Observability
    trace_id: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)

    # Idempotency and tracing
    client_msg_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    request_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("llm_requests.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
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
    conversation: Mapped[Conversation] = relationship(
        "Conversation",
        back_populates="messages",
        foreign_keys=[conversation_id],
    )
    llm_request: Mapped[LLMRequest | None] = relationship(
        "LLMRequest",
        back_populates="messages",
        foreign_keys=[request_id],
    )

    # Constraints
    __table_args__ = (
        UniqueConstraint("conversation_id", "seq", name="messages_conversation_id_seq_key"),
        UniqueConstraint(
            "conversation_id", "client_msg_id", name="messages_conversation_id_client_msg_id_key"
        ),
    )
