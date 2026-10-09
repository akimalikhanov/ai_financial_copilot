"""Answer-quality signals derived after streaming: the retrieval-confidence badge and
the uncited-fact share. Both feed the trace; the badge also feeds the SSE `metadata` event."""

import re
from collections.abc import Sequence

from src.schemas.retrieval import AnswerCitationSpan

# A fact-bearing number: money, a percentage, a decimal, or three or more digits. A comma
# counts only as a thousands separator, so "2020," stays a year.
_FACT_RE = re.compile(
    r"\$[\d,.]*\d|\d+(?:\.\d+)?%|(?:\d{1,3}(?:,\d{3})+|\d{3,})(?:\.\d+)?|\d+\.\d+"
)
# A year alone ("2022", "FY2022") dates a sentence but states no fact.
_YEAR_RE = re.compile(r"(?:19|20)\d{2}")
# A sentence ends at . ! ? before whitespace or the end, or at a line break, so "$4.2"
# and "15.5%" stay whole and each bullet is its own sentence. "vs.", "e.g." and "i.e."
# don't end one.
_SENTENCE_RE = re.compile(r"\S[^\n]*?(?:(?<!\bvs)(?<!\be\.g)(?<!\bi\.e)[.!?](?=\s|$)|(?=\n)|$)")
# Where a block ends: a blank line, or a line opening a list item, table row or heading.
# A prose paragraph is one block; each list item and table row is its own.
_BLOCK_END_RE = re.compile(r"\n(?=[ \t]*(?:\n|[-*+] |\d+[.)] |\||#))")


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


def _states_fact(sentence: str) -> bool:
    return any(not _YEAR_RE.fullmatch(m.group()) for m in _FACT_RE.finditer(sentence))


def uncited_fact_share(text: str, spans: Sequence[AnswerCitationSpan]) -> float | None:
    """Share of fact-bearing sentences with no citation marker after them in their block;
    None when the answer states no fact.

    A marker covers the sentences before it in the same block, so one marker closing a
    paragraph covers the paragraph, while each list item and table row needs its own.
    Headings and lead-in lines ending in ":" are skipped.

    Grounding is measured against the parsed spans, not the text: `text` is the
    citation-*stripped* answer, so the `[Sn]` markers a regex would look for have
    already been removed by the parser and could never match. A span's `end` is where
    its marker stood; a span's extent is not used, since it runs back to the previous
    marker and one trailing marker would otherwise cover the whole answer.
    """
    markers = [s.end for s in spans]
    facts = uncited = 0
    for m in _SENTENCE_RE.finditer(text):
        sentence = m.group()
        if sentence.startswith("#") or sentence.rstrip(" *").endswith(":"):
            continue
        if not _states_fact(sentence):
            continue
        facts += 1
        block_end = _BLOCK_END_RE.search(text, m.end())
        end = block_end.start() if block_end else len(text)
        if not any(m.start() < p <= end for p in markers):
            uncited += 1
    return uncited / facts if facts else None
