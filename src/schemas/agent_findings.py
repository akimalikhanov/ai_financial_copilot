from __future__ import annotations

import string
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

_LEADING_ARTICLES = ("the ", "a ", "an ")
# Separator between an observation's aspect and its item in the ledger key. Chosen not to
# occur in a model-authored aspect key, which the prompt asks to be a short snake_case name.
_KEY_SEP = "::"


def normalize_item_name(name: str) -> str:
    """D9's identity key for a `NamedItem`: case-fold, strip surrounding whitespace and
    punctuation, drop a leading article, collapse internal whitespace.

    Deterministic string handling only — no similarity computation (FR-12). Two names
    differing by more than case, surrounding punctuation, leading article, or internal
    spacing are simply different items; canonical-name reuse is a prompt-enforced
    discipline (D9, EC-5), not something this function tries to recover.

    Lives here rather than in `gates.py` because both the gates and the `FindingsLedger`
    key on it, and the field it keys is defined in this module.
    """
    key = " ".join(name.split()).strip().casefold()
    key = key.strip(string.punctuation + string.whitespace)
    for article in _LEADING_ARTICLES:
        if key.startswith(article):
            key = key[len(article) :]
            break
    return " ".join(key.split())


def observation_key(observation: Observation) -> str:
    """The `FindingsLedger` identity for one observation: aspect, plus the item it names.

    Keying on `aspect` alone silently collapsed distinct items (P1-1). The prompt requires
    one item per observation but never one aspect per item, so three segments reported as
    three observations sharing `segment_performance` were three in and one out — and a
    discovered set of segments is the dominant shape for the sequential-depth mechanism.

    Update-in-place still works across the unresolved -> resolved transition, because the
    item's name does not change when its value is found.
    """
    if observation.named_item is None:
        return observation.aspect
    item = normalize_item_name(observation.named_item.name)
    return f"{observation.aspect}{_KEY_SEP}{item}" if item else observation.aspect


def describe_key(key: str) -> str:
    """Render a ledger key back into something a model-facing message can quote.

    The composite key is internal — a rejection must never hand it back as if it were an
    `aspect` the model should emit verbatim.
    """
    aspect, sep, item = key.partition(_KEY_SEP)
    return f'{aspect} (item: "{item}")' if sep else aspect


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
