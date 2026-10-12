"""Unit tests for the number-grounding matcher (Pattern 4a, number half)."""

from __future__ import annotations

from src.services.chat.agent.number_grounding import (
    NumberGrounding,
    strip_marks,
    verify_marked,
    verify_value,
)


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

    def test_stated_scale_overrides_finding_unit(self) -> None:
        text = "(in millions, except per share data) | Revenue | $ 68 |"
        assert verify_value(68, "M", [text]) == NumberGrounding.GROUNDED
        assert verify_value(68, "B", [text]) == NumberGrounding.NOT_FOUND

    def test_stated_thousands_scale_matches_millions_finding(self) -> None:
        assert (
            verify_value(68, "M", ["(in thousands) | Revenue | 68,000 |"])
            == NumberGrounding.GROUNDED
        )

    def test_nil_dash_cell_grounds_zero(self) -> None:
        text = "(in millions) | Revenue | $ 68 | $ 82 | $ - |"
        assert verify_value(0, "M", [text]) == NumberGrounding.GROUNDED

    def test_markdown_separator_is_not_a_nil_cell(self) -> None:
        assert verify_value(0, "M", ["| Revenue | 2022 |\n|---|---|"]) == NumberGrounding.NOT_FOUND

    def test_nil_dash_in_prose_grounds_zero(self) -> None:
        text = "recognized collaboration revenue of $ 68 million, $82 million and $-, respectively."
        assert verify_value(0, "M", [text]) == NumberGrounding.GROUNDED

    def test_negative_amount_is_not_a_nil_dash(self) -> None:
        assert verify_value(0, "M", ["a loss of $-5 million and $ -0.4 million"]) == (
            NumberGrounding.NOT_FOUND
        )


class TestVerifyMarked:
    def test_only_marked_numbers_are_checked(self) -> None:
        # The unmarked year and note number are prose, not asserted magnitudes.
        claim = "In 2022 margin fell <n>4pp</n> (Note 14)."
        assert verify_marked(claim, ["margin down 4 percentage points"]) == [
            ("4pp", NumberGrounding.GROUNDED)
        ]

    def test_scale_word_in_the_mark_is_applied(self) -> None:
        texts = ["(in millions) | Cost of sales | 1,500 |"]
        assert verify_marked("a <n>$1.5 billion</n> rise", texts) == [
            ("$1.5 billion", NumberGrounding.GROUNDED)
        ]

    def test_unlocated_number_is_not_found(self) -> None:
        assert verify_marked("rose <n>12%</n>", ["rose 9%"]) == [("12%", NumberGrounding.NOT_FOUND)]

    def test_mark_without_a_number_is_unverifiable(self) -> None:
        assert verify_marked("<n>several</n>", ["text"]) == [
            ("several", NumberGrounding.UNVERIFIABLE)
        ]

    def test_strip_marks_keeps_the_numbers(self) -> None:
        assert strip_marks("rose <n>12%</n> to <n>$3m</n>") == "rose 12% to $3m"
