"""Unit tests for confidence scoring and citation-span grounding.

`has_ungrounded_claims` runs on the citation-*stripped* answer, so grounding must be
measured against the parsed spans' char offsets — the old `[Sn]` regex could never match
text the parser had already stripped, making every numeric sentence look ungrounded.
"""

from __future__ import annotations

from src.schemas.retrieval import AnswerCitationSpan
from src.services.chat.confidence import compute_confidence, has_ungrounded_claims


class TestComputeConfidence:
    def test_no_chunks_is_none(self) -> None:
        assert compute_confidence(0.9, 0) == "none"

    def test_missing_score_is_medium(self) -> None:
        assert compute_confidence(None, 3) == "medium"

    def test_score_bands(self) -> None:
        assert compute_confidence(0.7, 1) == "high"
        assert compute_confidence(0.25, 1) == "medium"
        assert compute_confidence(0.24, 1) == "low"


class TestHasUngroundedClaims:
    def test_cited_numeric_sentence_is_grounded(self) -> None:
        text = "Revenue was $1,200 million in 2023"
        spans = [AnswerCitationSpan(start=0, end=len(text), ref_ids=("S1",))]
        assert has_ungrounded_claims(text, spans) is False

    def test_uncited_numeric_sentence_is_ungrounded(self) -> None:
        text = "Revenue was $1,200 million in 2023"
        assert has_ungrounded_claims(text, []) is True

    def test_empty_spans_over_numeric_text_is_ungrounded(self) -> None:
        assert has_ungrounded_claims("Margins fell 12% year over year.", []) is True

    def test_non_numeric_text_is_never_ungrounded(self) -> None:
        assert has_ungrounded_claims("Margins fell sharply.", []) is False

    def test_second_sentence_uncited_is_detected(self) -> None:
        # Only the first sentence is covered; the second carries a fact and is not.
        text = "Revenue was $100. Costs rose 12% though."
        spans = [AnswerCitationSpan(start=0, end=17, ref_ids=("S1",))]
        assert has_ungrounded_claims(text, spans) is True

    def test_all_sentences_cited_is_grounded(self) -> None:
        text = "Revenue was $100. Costs rose 12% though."
        spans = [
            AnswerCitationSpan(start=0, end=17, ref_ids=("S1",)),
            AnswerCitationSpan(start=18, end=len(text), ref_ids=("S2",)),
        ]
        assert has_ungrounded_claims(text, spans) is False
