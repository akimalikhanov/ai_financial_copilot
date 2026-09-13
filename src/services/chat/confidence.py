"""Answer-quality signals derived after streaming: the retrieval-confidence badge and
the ungrounded-claim guardrail flag. Both feed the SSE `metadata` event and the trace."""

import re
from collections.abc import Sequence

from src.schemas.retrieval import AnswerCitationSpan

_FACT_PATTERN = re.compile(r"\$[\d,.]+|\d+\.?\d*%|\b(19|20)\d{2}\b|\d[\d,]{2,}")


def compute_confidence(
    top_score: float | None, num_chunks: int, *, scores_are_rerank: bool = True
) -> str:
    """Retrieval confidence from the top chunk's cross-encoder score.

    The thresholds below are calibrated on cross-encoder scores (~0–1). When reranking
    fell open or is switched off the chunks instead carry RRF fusion scores (~0.05 at
    k=20), which would land under every threshold and report a *ranking* outage as weak
    grounding — a system fault disguised as a corpus one. Unknown, not low, is the honest
    answer there; the degradation itself is surfaced separately.
    """
    if num_chunks == 0:
        return "none"
    if top_score is None or not scores_are_rerank:
        return "medium"
    if top_score >= 0.7:
        return "high"
    if top_score >= 0.25:
        return "medium"
    return "low"


def has_ungrounded_claims(text: str, spans: Sequence[AnswerCitationSpan]) -> bool:
    """True if any fact-bearing sentence is covered by no citation span.

    Grounding is measured against the parsed spans, not the text: `text` is the
    citation-*stripped* answer, so the `[Sn]` markers a regex would look for have
    already been removed by the parser and could never match.
    """
    offset = 0
    for sentence in text.split("."):
        start, end = offset, offset + len(sentence)
        offset = end + 1  # +1 for the "." consumed by split
        if not _FACT_PATTERN.search(sentence):
            continue
        if not any(s.start < end and s.end > start for s in spans):
            return True
    return False
