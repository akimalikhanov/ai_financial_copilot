from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class EntityFinding(BaseModel):
    model_config = ConfigDict(frozen=True)

    entity: str
    available: bool
    value: float | None = None
    currency: str | None = None
    period_end: str | None = None
    source_chunks: list[str] = Field(
        default=[],
        description='Excerpt IDs from search results that contain the value, exactly as shown (e.g. ["S3", "S7"]).',
    )
    reason: str | None = None
    unit: str | None = Field(
        default=None,
        description="Scale suffix as stated in the document: 'M' for millions, 'B' for billions, 'K' for thousands, '' for absolute values.",
    )


class AgentFindings(BaseModel):
    model_config = ConfigDict(frozen=True)

    metric_requested: str
    findings: tuple[EntityFinding, ...]
    comparison_op: Literal["argmin", "argmax", "list", "none"] | None = None


class Observation(BaseModel):
    model_config = ConfigDict(frozen=True)

    claim: str
    evidence_chunks: list[str] = Field(
        description='Excerpt IDs from search results that support this claim, exactly as shown (e.g. ["S3", "S7"]).',
    )
    confidence: Literal["high", "medium", "low"]
    refuted_by: list[str] | None = Field(
        default=None,
        description="Excerpt IDs that contradict this claim, exactly as shown in search results.",
    )


class AnalyticalFindings(BaseModel):
    model_config = ConfigDict(frozen=True)

    question: str
    observations: tuple[Observation, ...]
    conclusion: str | None = None
    gaps: list[str] | None = None
