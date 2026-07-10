from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import ARRAY, JSON, DateTime, Integer, Numeric, Text, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.base import Base

if TYPE_CHECKING:
    from src.models.canary_run_result import CanaryRunResult


class CanaryRun(Base):
    """One row per canary/eval RunOutput (manifest + aggregate metrics)."""

    __tablename__ = "canary_runs"

    id: Mapped[UUID] = mapped_column(primary_key=True, server_default="gen_random_uuid()")

    run_kind: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("'agentic'"))
    run_description: Mapped[str | None] = mapped_column(Text, nullable=True)

    run_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    git_sha: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    test_set: Mapped[str | None] = mapped_column(Text, nullable=True)
    test_set_hash: Mapped[str | None] = mapped_column(Text, nullable=True)
    model: Mapped[str | None] = mapped_column(Text, nullable=True)
    reasoning_effort: Mapped[str | None] = mapped_column(Text, nullable=True)
    max_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    judge_model: Mapped[str | None] = mapped_column(Text, nullable=True)
    k_values: Mapped[list[int] | None] = mapped_column(ARRAY(Integer), nullable=True)
    total_questions: Mapped[int | None] = mapped_column(Integer, nullable=True)
    evaluated: Mapped[int | None] = mapped_column(Integer, nullable=True)
    excluded: Mapped[list] = mapped_column(JSON, nullable=False, server_default=text("'[]'::jsonb"))

    retrieval: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    correctness: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    correctness_overall: Mapped[float | None] = mapped_column(nullable=True)
    judge: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    hallucination: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    hallucination_rate_mean: Mapped[float | None] = mapped_column(nullable=True)

    total_cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(12, 6), nullable=True)
    total_latency_s: Mapped[float | None] = mapped_column(nullable=True)

    regressions: Mapped[list] = mapped_column(
        JSON, nullable=False, server_default=text("'[]'::jsonb")
    )
    raw_manifest: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default="now()")
    updated_at: Mapped[datetime] = mapped_column(nullable=False, server_default="now()")

    results: Mapped[list[CanaryRunResult]] = relationship(
        "CanaryRunResult",
        back_populates="canary_run",
        cascade="all, delete-orphan",
    )
