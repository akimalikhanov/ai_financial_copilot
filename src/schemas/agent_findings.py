from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class EntityFinding:
    entity: str
    available: bool
    value: float | None = None
    currency: str | None = None
    period_end: str | None = None
    source_chunks: list[str] | None = None
    reason: str | None = None
    unit: str | None = None  # scale suffix as stated in document: "M", "B", "K", or "" for absolute


@dataclass(frozen=True, slots=True)
class AgentFindings:
    metric_requested: str
    findings: tuple[EntityFinding, ...]
    comparison_op: Literal["argmin", "argmax", "list", "none"] | None = None


@dataclass(frozen=True, slots=True)
class Observation:
    claim: str
    evidence_chunks: list[str]
    confidence: Literal["high", "medium", "low"]
    refuted_by: list[str] | None = None


@dataclass(frozen=True, slots=True)
class AnalyticalFindings:
    question: str
    observations: tuple[Observation, ...]
    conclusion: str | None = None
    gaps: list[str] | None = None
