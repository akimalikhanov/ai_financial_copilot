"""Unit tests for the number-grounding matcher (Pattern 4a, number half)."""

from __future__ import annotations

from src.services.chat.agent.number_grounding import NumberGrounding, verify_value


class TestVerifyValue:
    def test_thousands_separator_matches(self) -> None:
        assert verify_value(1234.5, "M", ["Total revenue was 1,234.5"]) == NumberGrounding.GROUNDED

    def test_adjacent_scale_word_matches(self) -> None:
        assert (
            verify_value(1234.5, "M", ["revenue of $1,234.5 million"]) == NumberGrounding.GROUNDED
        )

    def test_cross_scale_matches(self) -> None:
        assert verify_value(1.2, "B", ["1,200.0 million"]) == NumberGrounding.GROUNDED

    def test_parenthesized_negative_matches(self) -> None:
        assert verify_value(1234.5, "M", ["(1,234.5)"]) == NumberGrounding.GROUNDED

    def test_within_rounding_tolerance_matches(self) -> None:
        assert verify_value(1235.0, "M", ["1,234.5"]) == NumberGrounding.GROUNDED

    def test_outside_tolerance_not_found(self) -> None:
        assert verify_value(1300.0, "M", ["1,234.5"]) == NumberGrounding.NOT_FOUND

    def test_no_numbers_at_all_not_found(self) -> None:
        assert (
            verify_value(1234.5, "M", ["revenue increased materially"]) == NumberGrounding.NOT_FOUND
        )

    def test_none_value_unverifiable(self) -> None:
        assert verify_value(None, "M", ["1,234.5"]) == NumberGrounding.UNVERIFIABLE

    def test_empty_texts_unverifiable(self) -> None:
        assert verify_value(1234.5, "M", []) == NumberGrounding.UNVERIFIABLE

    def test_table_row_with_multiple_tokens_matches(self) -> None:
        assert (
            verify_value(1234.5, "M", ["Revenue 1,234.5 987.6 1,102.3"]) == NumberGrounding.GROUNDED
        )
