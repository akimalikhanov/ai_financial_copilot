from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Figure(BaseModel):
    model_config = ConfigDict(frozen=True)

    metric: str = Field(description='The metric as the question names it, e.g. "revenue".')
    amount: float = Field(description="The number exactly as printed, in the document's own scale.")
    unit: Literal["", "K", "M", "B"] | None = Field(
        description=(
            'Scale stated in the document: "K" thousands, "M" millions, "B" billions, "" '
            "absolute figures. null when the document does not state a scale."
        ),
    )
    currency: str | None = Field(description="ISO 4217 code, e.g. USD. null if not monetary.")
    period_end: str | None = Field(
        description=(
            "Period end date as YYYY-MM-DD, copied from the column header. null when the "
            "header gives no date; never infer one from a year."
        ),
    )
    fiscal_label: str | None = Field(
        description='The period as printed, e.g. "FY2023" or "52 weeks ended 30 Sep 2023".',
    )


class Finding(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str = Field(
        description=(
            'The key shown in brackets in the search result, e.g. "Acme Corp" or "A2". '
            "Never invent one."
        ),
    )
    claim: str = Field(
        description=(
            "One sentence stating one claim. A negative is its own finding with supported: false."
        ),
    )
    supported: bool = Field(
        description=(
            "False when you searched this key and the documents do not say — state the "
            "absence in claim and leave evidence and figures empty."
        ),
    )
    evidence: list[str] = Field(
        description='Excerpt IDs that support the claim, exactly as shown (e.g. ["S3", "S7"]).',
    )
    confidence: Literal["high", "medium", "low"]
    figures: list[Figure] = Field(
        default=[],
        description="Numeric values behind the claim, one per metric and period. Empty if none.",
    )


class FindingsReport(BaseModel):
    """`report_findings` arguments."""

    model_config = ConfigDict(frozen=True)

    findings: tuple[Finding, ...]
    comparison_op: Literal["argmin", "argmax", "list", "none"] | None = None
    conclusion: str | None = None


class AgentFindings(FindingsReport):
    """What a run serves: every report folded together, plus one line per plan key that
    never produced a finding."""

    unresolved: tuple[str, ...] = ()
