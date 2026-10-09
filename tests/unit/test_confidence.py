"""Unit tests for confidence scoring and citation-span grounding.

`uncited_fact_share` runs on the citation-*stripped* answer, so grounding must be
measured against the parsed spans' char offsets — the old `[Sn]` regex could never match
text the parser had already stripped, making every numeric sentence look ungrounded.
"""

from __future__ import annotations

from src.schemas.retrieval import AnswerCitationSpan
from src.services.chat.confidence import compute_confidence, uncited_fact_share


class TestComputeConfidence:
    def test_no_chunks_is_none(self) -> None:
        assert compute_confidence(0.9, 0) == "none"

    def test_missing_score_is_medium(self) -> None:
        assert compute_confidence(None, 3) == "medium"

    def test_score_bands(self) -> None:
        assert compute_confidence(0.7, 1) == "high"
        assert compute_confidence(0.25, 1) == "medium"
        assert compute_confidence(0.24, 1) == "low"

    def test_fusion_scale_scores_are_unknown_not_low(self) -> None:
        """A reranker outage must not be reported as weak grounding.

        RRF scores sit ~0.05 (k=20), under every band below, so applying the
        cross-encoder thresholds to them turns a ranking outage into a corpus verdict.
        """
        assert compute_confidence(0.05, 3, scores_are_rerank=False) == "medium"
        assert compute_confidence(0.05, 3) == "low"

    def test_no_chunks_still_none_regardless_of_scale(self) -> None:
        assert compute_confidence(0.05, 0, scores_are_rerank=False) == "none"


def _marker_at(end: int) -> AnswerCitationSpan:
    """A span whose `[Sn]` marker stood at `end` in the clean text."""
    return AnswerCitationSpan(start=0, end=end, ref_ids=("S1",))


class TestUncitedFactShare:
    def test_cited_numeric_sentence_is_grounded(self) -> None:
        text = "Revenue was $1,200 million in 2023"
        assert uncited_fact_share(text, [_marker_at(len(text))]) == 0.0

    def test_uncited_numeric_sentence_is_ungrounded(self) -> None:
        assert uncited_fact_share("Revenue was $1,200 million in 2023", []) == 1.0

    def test_empty_spans_over_numeric_text_is_ungrounded(self) -> None:
        assert uncited_fact_share("Margins fell 12% year over year.", []) == 1.0

    def test_text_without_facts_has_no_share(self) -> None:
        assert uncited_fact_share("Margins fell sharply.", []) is None

    def test_year_alone_is_not_a_fact(self) -> None:
        assert uncited_fact_share("In FY2022 margins fell sharply.", []) is None

    def test_year_before_a_comma_is_not_a_fact(self) -> None:
        text = "In 2020, there was no revenue. On December 31, 2022, the plan ended."
        assert uncited_fact_share(text, []) is None

    def test_thousands_separator_is_a_fact(self) -> None:
        assert uncited_fact_share("Headcount reached 2,890.", []) == 1.0

    def test_second_sentence_uncited_is_detected(self) -> None:
        # The marker stands between the sentences, so it covers only the first.
        text = "Revenue was $100. Costs rose 12% though."
        assert uncited_fact_share(text, [_marker_at(17)]) == 0.5

    def test_decimal_point_does_not_split_sentence(self) -> None:
        # Marker after the period: "$4.2 billion" is one sentence, not "$4" + "2 billion".
        text = "Revenue was $4.2 billion in 2022."
        assert uncited_fact_share(text, [_marker_at(len(text))]) == 0.0

    def test_vs_does_not_split_sentence(self) -> None:
        text = "Expense rose to ¥9,642m vs. ¥9,603m a year earlier."
        assert uncited_fact_share(text, [_marker_at(len(text))]) == 0.0

    def test_marker_closing_a_paragraph_covers_it(self) -> None:
        text = "Revenue was $4.2 billion. Margin was 15.5%."
        assert uncited_fact_share(text, [_marker_at(len(text))]) == 0.0

    def test_marker_does_not_cover_an_earlier_paragraph(self) -> None:
        text = "Revenue was $4.2 billion.\n\nMargin was 15.5%."
        assert uncited_fact_share(text, [_marker_at(len(text))]) == 0.5

    def test_each_bullet_needs_its_own_marker(self) -> None:
        text = "Revenue by year:\n- 2021: $82 million\n- 2022: $68 million"
        assert uncited_fact_share(text, [_marker_at(len(text))]) == 0.5

    def test_each_table_row_needs_its_own_marker(self) -> None:
        text = "| Year | Revenue |\n|---|---|\n| 2022 | $68M |\n| 2021 | $82M |"
        assert uncited_fact_share(text, [_marker_at(len(text))]) == 0.5

    def test_headings_and_lead_ins_are_skipped(self) -> None:
        text = "## Revenue 2020-2022\n\nRevenue for FY2022 was:\n- $68 million"
        assert uncited_fact_share(text, [_marker_at(len(text))]) == 0.0

    def test_all_sentences_cited_is_grounded(self) -> None:
        text = "Revenue was $100. Costs rose 12% though."
        spans = [
            AnswerCitationSpan(start=0, end=17, ref_ids=("S1",)),
            AnswerCitationSpan(start=18, end=len(text), ref_ids=("S2",)),
        ]
        assert uncited_fact_share(text, spans) == 0.0
