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


class NamedItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str = Field(
        description="The item's label exactly as the documents name it (e.g. 'Payments segment', 'Acme Sub Ltd'). Reuse the exact same name for the same item on every turn of this run.",
    )
    status: Literal["unresolved", "resolved", "confirmed_absent"] = Field(
        description="'unresolved' — the item is named but its value has not been found yet. 'resolved' — the value was found and is stated in this observation's claim. 'confirmed_absent' — the documents do not disclose this value at all, which is not the same as 'not yet found'.",
    )


class Observation(BaseModel):
    model_config = ConfigDict(frozen=True)

    aspect: str = Field(
        description="A short, stable key naming what this observation is about (e.g. 'revenue_driver', 'margin_trend') — the same aspect across turns updates the same conclusion.",
    )
    claim: str
    evidence_chunks: list[str] = Field(
        description='Excerpt IDs from search results that support this claim, exactly as shown (e.g. ["S3", "S7"]).',
    )
    confidence: Literal["high", "medium", "low"]
    refuted_by: list[str] | None = Field(
        default=None,
        description="Excerpt IDs that contradict this claim, exactly as shown in search results.",
    )
    named_item: NamedItem | None = Field(
        default=None,
        description="The one specific item (segment, subsidiary, transaction) this observation is about, when it names such an item. At most one item per observation — if you have gaps on several items, split them across separate observations.",
    )


class AnalyticalFindings(BaseModel):
    model_config = ConfigDict(frozen=True)

    question: str
    observations: tuple[Observation, ...]
    conclusion: str | None = None
    gaps: list[str] | None = None
