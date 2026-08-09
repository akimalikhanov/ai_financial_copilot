"""Deterministic number-grounding: does the cited chunk text contain the asserted magnitude?

Pattern 4a's number half. The label half (does the ref resolve to a real chunk) is
`EvidenceLedger.resolve_refs` + `FindingsLedger._is_grounded`; this is the orthogonal
question those cannot answer — a correctly-resolved citation to a chunk that does not
state the number.

Advisory by construction: this module returns a verdict, never raises and never filters.
Enforcement would re-create the rejection path D3 deleted (see the plan doc, §2.1).
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from enum import StrEnum

UNIT_TO_MILLIONS: dict[str | None, float] = {
    "B": 1_000.0,
    "M": 1.0,
    "K": 0.001,
    "": 0.000_001,  # absolute / units
    None: 1.0,  # assume millions when unspecified
}


def to_millions(value: float, unit: str | None) -> float:
    """Scale value to millions for unit-safe comparison."""
    return value * UNIT_TO_MILLIONS.get(unit, 1.0)


class NumberGrounding(StrEnum):
    GROUNDED = "grounded"
    NOT_FOUND = "not_found"
    UNVERIFIABLE = "unverifiable"


_NUMBER_RE = re.compile(
    r"\(?\s*[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?\s*\)?|\(?\s*[-+]?\d+(?:\.\d+)?\s*\)?"
)
# Scale words expressed in millions, so they line up with UNIT_TO_MILLIONS.
_SCALE_WORDS: dict[str, float] = {
    "billion": 1_000.0,
    "billions": 1_000.0,
    "bn": 1_000.0,
    "b": 1_000.0,
    "million": 1.0,
    "millions": 1.0,
    "mn": 1.0,
    "m": 1.0,
    "thousand": 0.001,
    "thousands": 0.001,
    "k": 0.001,
}
_TRAILING_WINDOW = 20  # chars scanned after a number for an adjacent scale word

_REL_TOL = 0.005
_ABS_TOL = 1e-9


def _parse_number(token: str) -> float | None:
    negative = "(" in token
    cleaned = token.strip().strip("()").replace(",", "").strip()
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return -value if negative else value


def _candidates(raw: float, trailing: str, finding_unit: str | None) -> list[float]:
    """Magnitudes (in millions) a bare number token could plausibly denote."""
    out = [raw * UNIT_TO_MILLIONS.get("", 1.0)]  # literal, treated as an absolute value
    word_match = re.match(r"\s*([a-zA-Z]+)", trailing)
    if word_match:
        scale = _SCALE_WORDS.get(word_match.group(1).lower())
        if scale is not None:
            out.append(raw * scale)
    # The finding's own declared unit — covers a table whose scale sits in a header far
    # from the cell (e.g. "$ in millions" at the top of the table).
    out.append(to_millions(raw, finding_unit))
    return out


def _matches(a: float, b: float) -> bool:
    return math.isclose(abs(a), abs(b), rel_tol=_REL_TOL, abs_tol=_ABS_TOL)


def verify_value(
    value: float | None,
    unit: str | None,
    texts: Sequence[str],
) -> NumberGrounding:
    """Scan `texts` for a number matching `value` (scaled by `unit`) within tolerance."""
    if value is None or not texts:
        return NumberGrounding.UNVERIFIABLE

    target = to_millions(value, unit)
    for text in texts:
        for m in _NUMBER_RE.finditer(text):
            raw = _parse_number(m.group())
            if raw is None:
                continue
            trailing = text[m.end() : m.end() + _TRAILING_WINDOW]
            if any(_matches(target, c) for c in _candidates(raw, trailing, unit)):
                return NumberGrounding.GROUNDED
    return NumberGrounding.NOT_FOUND
