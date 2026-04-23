from __future__ import annotations

import contextlib
import re
from typing import Literal

from rapidfuzz import fuzz

Kind = Literal["number", "boolean", "name", "names"]

_REFUSAL_RE = re.compile(
    r"\b(n/a|not available|not mentioned|not disclosed|not found|not stated"
    r"|no information|does not mention|does not disclose|does not (name|specify|include|provide|detail|list|report|contain|discuss)"
    r"|do not (name|specify|include|provide|detail|list|report|contain|discuss)"
    r"|cannot find|insufficient|no data"
    r"|excerpts? do not|report does not|annual report does not"
    r"|no (specific|such|relevant|further)|not (specified|provided|reported|included|detailed|listed))\b",
    re.IGNORECASE,
)

_NUM_RE = re.compile(r"-?[$£€]?\s*(\d[\d,]*(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*%?")

_BOOL_RE = re.compile(r"\b(true|false|yes|no)\b", re.IGNORECASE)

_BOOL_MAP = {"yes": "true", "no": "false", "true": "true", "false": "false"}

_NAME_LINE_SPLIT_RE = re.compile(r"[,;\n]")  # split on these; NOT on "and" — titles contain it
_NAME_CLEANUP_RE = re.compile(
    r"^[\s\-\*•]+|[\s\.\-]+$"
)  # strip bullet prefixes and trailing punctuation
_PAREN_RE = re.compile(r"\s*\(.*?\)")  # strip parenthetical suffixes e.g. "(CEO)"

FUZZY_NAME_THRESHOLD = 85


def _parse_all_numbers(text: str) -> list[float]:
    text = text.replace(",", "")
    results = []
    for m in _NUM_RE.finditer(text):
        with contextlib.suppress(ValueError):
            results.append(float(m.group(1)))
    return results


def _numbers_match(got: float, expected: float) -> bool:
    abs_tol = max(0.01, abs(expected) * 0.01)
    return abs(got - expected) <= abs_tol


def _score_number(answer: str, expected: list[str]) -> dict:
    candidates = _parse_all_numbers(answer)
    if not candidates:
        return {"correct": False, "reason": "no_number_found"}
    for exp_str in expected:
        exp_candidates = _parse_all_numbers(exp_str)
        for exp in exp_candidates:
            if any(_numbers_match(got, exp) for got in candidates):
                return {"correct": True, "reason": "number_match"}
    return {"correct": False, "reason": "number_mismatch"}


def _score_boolean(answer: str, expected: list[str]) -> dict:
    m = _BOOL_RE.search(answer)
    if m is None:
        return {"correct": False, "reason": "no_boolean_found"}
    got = _BOOL_MAP[m.group(1).lower()]
    for exp_str in expected:
        em = _BOOL_RE.search(exp_str)
        if em and _BOOL_MAP[em.group(1).lower()] == got:
            return {"correct": True, "reason": "boolean_match"}
    return {"correct": False, "reason": "boolean_mismatch"}


def _score_name(answer: str, expected: list[str]) -> dict:
    # Try full-string match first, then check if expected name appears within the answer
    for exp in expected:
        exp_clean = exp.strip()
        if fuzz.ratio(answer.strip(), exp_clean) >= FUZZY_NAME_THRESHOLD:
            return {"correct": True, "reason": "name_match"}
        if fuzz.partial_ratio(exp_clean, answer) >= FUZZY_NAME_THRESHOLD:
            return {"correct": True, "reason": "name_match"}
    return {"correct": False, "reason": "name_mismatch"}


def _score_names(answer: str, expected: list[str]) -> dict:
    raw_candidates = [t for t in _NAME_LINE_SPLIT_RE.split(answer) if t.strip()]
    candidates = []
    for c in raw_candidates:
        c = _PAREN_RE.sub("", c)
        c = _NAME_CLEANUP_RE.sub("", c).strip()
        if c:
            candidates.append(c)
    # Split expected on comma (e.g. "Chief Legal Counsel,Chief Financial Officer")
    gold_raw = ",".join(e.strip() for e in expected if e.strip())
    gold = [g.strip() for g in gold_raw.split(",") if g.strip()]
    if not gold:
        return {"correct": False, "reason": "no_expected_names"}

    def fuzzy_match(name: str, pool: list[str]) -> bool:
        return any(
            fuzz.ratio(name, g) >= FUZZY_NAME_THRESHOLD
            or fuzz.partial_ratio(name, g) >= FUZZY_NAME_THRESHOLD
            for g in pool
        )

    tp = sum(1 for g in gold if fuzzy_match(g, candidates))
    recall = tp / len(gold)
    # Use recall-only: gold set may be incomplete so extra candidates are not penalized
    correct = recall >= 0.5
    return {"correct": correct, "reason": f"names_recall={recall:.2f}"}


def _score_na(answer: str) -> dict:
    correct = bool(_REFUSAL_RE.search(answer))
    return {"correct": correct, "reason": "na_refusal" if correct else "na_not_refused"}


def score_correctness(answer: str, kind: Kind, expected_answers: list[str]) -> dict:
    """Return {correct: bool, reason: str}."""
    if expected_answers == ["N/A"]:
        return _score_na(answer)
    dispatch = {
        "number": _score_number,
        "boolean": _score_boolean,
        "name": _score_name,
        "names": _score_names,
    }
    return dispatch[kind](answer, expected_answers)
