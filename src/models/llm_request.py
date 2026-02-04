from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import JSON, ForeignKey, Integer, Numeric, Text, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.base import Base

if TYPE_CHECKING:
    from src.models.message import Message


class LLMRequest(Base):
    """
    LLM request model representing a single LLM API call.

    Maps to the llm_requests table in PostgreSQL.
    """

    __tablename__ = "llm_requests"

    # Primary key
    id: Mapped[UUID] = mapped_column(primary_key=True, server_default="gen_random_uuid()")

    # Foreign keys
    conversation_id: Mapped[UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # LLM provider/model info
    provider: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(Text, nullable=False)

    # Request parameters
    request_params: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )

    # Token usage and stats
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reasoning_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Cost and performance
    cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(12, 6), nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ttft_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tps: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Error info
    error_code: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

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
        back_populates="llm_request",
    )
