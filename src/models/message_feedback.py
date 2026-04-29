from __future__ import annotations

import enum
from datetime import datetime
from uuid import UUID

from sqlalchemy import Enum as SQLEnum
from sqlalchemy import ForeignKey, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base


class FeedbackRating(str, enum.Enum):
    up = "up"
    down = "down"


class MessageFeedback(Base):
    __tablename__ = "message_feedback"

    id: Mapped[UUID] = mapped_column(primary_key=True, server_default="gen_random_uuid()")
    message_id: Mapped[UUID] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    rating: Mapped[FeedbackRating] = mapped_column(
        SQLEnum(FeedbackRating, native_enum=True, name="feedback_rating"),
        nullable=False,
    )
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default="now()")
    updated_at: Mapped[datetime] = mapped_column(nullable=False, server_default="now()")

    __table_args__ = (
        UniqueConstraint("message_id", "user_id", name="message_feedback_message_user_key"),
    )
