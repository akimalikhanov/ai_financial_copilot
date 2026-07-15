from __future__ import annotations

import contextlib
import json
import logging
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ValidationError
from rapidfuzz import fuzz

from src.services.llm_adapters.base_adapter import ChatMessage, Role
from src.services.llm_router import get_router
from src.utils.json_schema import build_response_format

logger = logging.getLogger(__name__)

Kind = Literal["number", "boolean", "name", "names", "drivers"]

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

# Matches "406.1 million", "2.5 billion", "350K", "1.2B", "500M" etc.
_PROSE_NUM_RE = re.compile(
    r"-?\$?(\d[\d,]*(?:\.\d+)?)\s*"
    r"(trillion|billion|million|thousand|[tbmk])\b",
    re.IGNORECASE,
)
_PROSE_SCALE = {
    "t": 1_000_000_000_000,
    "trillion": 1_000_000_000_000,
    "b": 1_000_000_000,
    "billion": 1_000_000_000,
    "m": 1_000_000,
    "million": 1_000_000,
    "k": 1_000,
    "thousand": 1_000,
}

_BOOL_RE = re.compile(r"\b(true|false|yes|no)\b", re.IGNORECASE)

_BOOL_MAP = {"yes": "true", "no": "false", "true": "true", "false": "false"}

_NAME_LINE_SPLIT_RE = re.compile(r"[,;\n]")  # split on these; NOT on "and" — titles contain it
_NAME_CLEANUP_RE = re.compile(
    r"^[\s\-\*•]+|[\s\.\-]+$"
)  # strip bullet prefixes and trailing punctuation
_PAREN_RE = re.compile(r"\s*\(.*?\)")  # strip parenthetical suffixes e.g. "(CEO)"

FUZZY_NAME_THRESHOLD = 85


def _parse_all_numbers(text: str) -> list[float]:
    results: list[float] = []
    # First pass: prose numbers with scale suffixes ("406.1 million", "2.5B")
    for m in _PROSE_NUM_RE.finditer(text):
        with contextlib.suppress(ValueError):
            base = float(m.group(1).replace(",", ""))
            scale = _PROSE_SCALE[m.group(2).lower()]
            results.append(base * scale)
    # Second pass: bare numbers (strip commas first)
    clean = _PROSE_NUM_RE.sub(" ", text).replace(",", "")
    for m in _NUM_RE.finditer(clean):
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
    if kind not in dispatch:
        # "drivers" has no deterministic scorer (name+magnitude in free text can't be
        # regexed reliably) — use the async LLM judge below (score_drivers_llm) instead.
        # Don't crash the run; the general LLM judge (metrics/judge.py) still runs and
        # produces faithfulness/relevance signal.
        return {"correct": False, "reason": "no_scorer_for_kind"}
    return dispatch[kind](answer, expected_answers)


_DRIVER_JUDGE_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "driver_recall_judge_v1.yaml"

_DRIVER_RECALL_THRESHOLD = 0.5


class _DriverHit(BaseModel):
    driver: str
    named: bool
    reason: str


class DriverRecallOutput(BaseModel):
    hits: list[_DriverHit]


def _load_driver_judge_prompt() -> str:
    with open(_DRIVER_JUDGE_PROMPT_PATH) as f:
        data = yaml.safe_load(f)
    return data["template"]


def _format_gold_drivers(expected_answers: list[str]) -> str:
    return "\n".join(f"{i + 1}. {d}" for i, d in enumerate(expected_answers))


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _parse_driver_judge_response(raw: str) -> DriverRecallOutput | None:
    text = raw.strip()
    m = _JSON_FENCE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        data = json.loads(text)
        return DriverRecallOutput.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as e:
        logger.warning("driver_judge_parse_failed: %s raw=%r", e, raw[:300])
        return None


async def score_drivers_llm(
    *,
    question: str,
    answer: str,
    expected_answers: list[str],
    model_id: str,
) -> dict:
    """LLM-judged recall over the hand-labeled driver list for kind='drivers'.

    Text-only comparison of MODEL_ANSWER against the gold driver list — does not verify
    that a named driver's citation lands in that driver's reference_pools page(s) (the
    other half of the doc's Stage 0.5 spec). That leg needs ref_id -> chunk -> page
    resolution, which isn't available for an offline re-score of an already-collected run
    (ref_id_to_chunk_id is unpopulated in analytical_tier1_baseline.json).

    Returns {correct: bool, reason: str} to match score_correctness's contract, with
    correct = recall >= 0.5 (same recall-threshold convention as _score_names).
    On LLM/parse failure, returns {correct: False, reason: "driver_judge_failed"}.
    """
    if not expected_answers:
        return {"correct": False, "reason": "no_expected_drivers"}

    system = _load_driver_judge_prompt()
    user = (
        f"QUESTION:\n{question}\n\n"
        f"GOLD_DRIVERS:\n{_format_gold_drivers(expected_answers)}\n\n"
        f"MODEL_ANSWER:\n{answer}"
    )
    messages = [
        ChatMessage(role=Role.system, content=system),
        ChatMessage(role=Role.user, content=user),
    ]
    response_format = build_response_format(
        "driver_recall_output", DriverRecallOutput.model_json_schema()
    )
    llm = get_router().get(model_id)
    try:
        resp = await llm.complete(
            messages=messages,
            temperature=0.0,
            response_format=response_format,
        )
    except Exception as e:
        logger.exception("driver_judge_llm_error: %s", e)
        return {"correct": False, "reason": "driver_judge_failed"}

    parsed = _parse_driver_judge_response(resp.text or "")
    if parsed is None or len(parsed.hits) != len(expected_answers):
        return {"correct": False, "reason": "driver_judge_failed"}

    n_hit = sum(1 for h in parsed.hits if h.named)
    recall = n_hit / len(expected_answers)
    correct = recall >= _DRIVER_RECALL_THRESHOLD
    missed = [h.driver for h in parsed.hits if not h.named]
    reason = f"driver_recall={recall:.2f} ({n_hit}/{len(expected_answers)} drivers named)"
    if missed:
        reason += f"; missed: {'; '.join(missed)}"
    return {"correct": correct, "reason": reason}
