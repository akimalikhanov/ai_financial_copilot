from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import JSON, Boolean, ForeignKey, Integer, Numeric, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.base import Base

if TYPE_CHECKING:
    from src.models.canary_run import CanaryRun


class CanaryRunResult(Base):
    """One row per PerQuestionResult within a canary_runs row."""

    __tablename__ = "canary_run_results"

    id: Mapped[UUID] = mapped_column(primary_key=True, server_default="gen_random_uuid()")

    canary_run_id: Mapped[UUID] = mapped_column(
        ForeignKey("canary_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    qid: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    question: Mapped[str | None] = mapped_column(Text, nullable=True)
    kind: Mapped[str | None] = mapped_column(Text, nullable=True)
    route: Mapped[str | None] = mapped_column(Text, nullable=True)
    excluded_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    retrieved_page_keys: Mapped[list | None] = mapped_column(JSON, nullable=True)
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    citation_spans: Mapped[list | None] = mapped_column(JSON, nullable=True)

    correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    correctness_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    judge: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    hallucination_rate: Mapped[float | None] = mapped_column(nullable=True)

    latency_s: Mapped[float | None] = mapped_column(nullable=True)
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(12, 6), nullable=True)

    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default="now()")

    canary_run: Mapped[CanaryRun] = relationship("CanaryRun", back_populates="results")
