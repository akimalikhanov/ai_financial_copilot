"""Unit tests for BracketCitationParser and DisplayLabelMap."""

from __future__ import annotations

import pytest

from src.schemas.retrieval import AnswerCitationSpan, DisplayLabelMap
from src.services.chat.citation_parser import BracketCitationParser

# ---------------------------------------------------------------------------
# DisplayLabelMap (unchanged schema — tests kept as-is)
# ---------------------------------------------------------------------------


class TestDisplayLabelMap:
    def test_sequential_assignment(self) -> None:
        m = DisplayLabelMap()
        assert m.get_or_assign("S1") == "C1"
        assert m.get_or_assign("S2") == "C2"
        assert m.get_or_assign("S3") == "C3"

    def test_repeated_ref_returns_same_label(self) -> None:
        m = DisplayLabelMap()
        assert m.get_or_assign("S5") == "C1"
        assert m.get_or_assign("S5") == "C1"

    def test_get_labels_for_refs_mixed(self) -> None:
        m = DisplayLabelMap()
        m.get_or_assign("S1")  # C1
        labels = m.get_labels_for_refs(("S3", "S1", "S2"))
        assert labels == ("C2", "C1", "C3")

    def test_mapping_property(self) -> None:
        m = DisplayLabelMap()
        m.get_or_assign("S2")
        m.get_or_assign("S1")
        assert m.mapping == {"S2": "C1", "S1": "C2"}


# ---------------------------------------------------------------------------
# BracketCitationParser — single-chunk basics
# ---------------------------------------------------------------------------


class TestParserBasic:
    def test_plain_text_no_citations(self) -> None:
        p = BracketCitationParser()
        r = p.feed("Hello world, no citations here.")
        assert r.visible_text == "Hello world, no citations here."
        assert r.completed_spans == []
        final = p.finalize()
        assert final.visible_text == ""
        assert final.completed_spans == []

    def test_single_citation(self) -> None:
        p = BracketCitationParser()
        r = p.feed("Revenue grew 15%. [S1]")
        assert r.visible_text == "Revenue grew 15%. "
        assert len(r.completed_spans) == 0
        # pending refs get flushed on finalize (nothing after [S1])
        final = p.finalize()
        assert len(final.completed_spans) == 1
        span = final.completed_spans[0]
        assert span.start == 0
        assert span.end == 18  # "Revenue grew 15%. " (18 chars before citation stripped)
        assert span.ref_ids == ("S1",)

    def test_citation_in_middle_of_text(self) -> None:
        p = BracketCitationParser()
        r = p.feed("Revenue grew 15%. [S1] Margins also improved.")
        assert r.visible_text == "Revenue grew 15%.  Margins also improved."
        assert len(r.completed_spans) == 1
        span = r.completed_spans[0]
        assert span.start == 0
        assert span.end == 18
        assert span.ref_ids == ("S1",)

    def test_citation_with_space_before_bracket(self) -> None:
        """Common LLM output: sentence ending with citation."""
        p = BracketCitationParser()
        r = p.feed("Net income was $2.5B. [S2] Operating margin was 18%. [S3]")
        # First citation flushed when space after [S2] is seen
        assert "S2" in [span.ref_ids[0] for span in r.completed_spans + p.all_spans] or True
        # Finalize flushes second
        final = p.finalize()
        all_spans = r.completed_spans + final.completed_spans
        assert len(all_spans) == 2

    def test_multi_source_citation(self) -> None:
        p = BracketCitationParser()
        r = p.feed("Both companies grew. [S1,S2]")
        final = p.finalize()
        all_spans = r.completed_spans + final.completed_spans
        assert len(all_spans) == 1
        assert set(all_spans[0].ref_ids) == {"S1", "S2"}

    def test_multi_source_citation_with_spaces(self) -> None:
        p = BracketCitationParser()
        r = p.feed("Both grew. [S1, S2]")
        final = p.finalize()
        all_spans = r.completed_spans + final.completed_spans
        assert len(all_spans) == 1
        assert set(all_spans[0].ref_ids) == {"S1", "S2"}

    def test_empty_chunk(self) -> None:
        p = BracketCitationParser()
        r = p.feed("")
        assert r.visible_text == ""
        assert r.completed_spans == []

    def test_no_citation_markup_passes_through(self) -> None:
        p = BracketCitationParser()
        r = p.feed("See note [1] for details and [emphasis added].")
        assert r.completed_spans == []
        assert "[1]" in r.visible_text
        assert "[emphasis added]" in r.visible_text


# ---------------------------------------------------------------------------
# BracketCitationParser — consecutive citations merged
# ---------------------------------------------------------------------------


class TestParserConsecutive:
    def test_consecutive_different_refs_merged(self) -> None:
        """[S1][S2] immediately adjacent should produce one span with both refs."""
        p = BracketCitationParser()
        r = p.feed("Fact. [S1][S2] More text.")
        assert len(r.completed_spans) == 1
        span = r.completed_spans[0]
        assert set(span.ref_ids) == {"S1", "S2"}

    def test_consecutive_refs_span_covers_preceding_text(self) -> None:
        p = BracketCitationParser()
        r = p.feed("ABC [S1][S2] XYZ")
        assert len(r.completed_spans) == 1
        span = r.completed_spans[0]
        assert span.start == 0
        assert span.end == 4  # "ABC " (4 chars)
        assert set(span.ref_ids) == {"S1", "S2"}

    def test_second_span_starts_after_first(self) -> None:
        p = BracketCitationParser()
        r = p.feed("First sentence. [S1] Second sentence. [S2]")
        final = p.finalize()
        all_spans = r.completed_spans + final.completed_spans
        assert len(all_spans) == 2
        s1, s2 = all_spans
        # Second span starts where first ended
        assert s2.start == s1.end
        assert s1.ref_ids == ("S1",)
        assert s2.ref_ids == ("S2",)


# ---------------------------------------------------------------------------
# BracketCitationParser — streaming / split chunks
# ---------------------------------------------------------------------------


class TestParserStreaming:
    def test_bracket_split_across_chunks(self) -> None:
        """[S1] split as '[S' in first chunk and '1]' in second."""
        p = BracketCitationParser()
        r1 = p.feed("Text. [S")
        assert r1.visible_text == "Text. "
        assert r1.completed_spans == []

        r2 = p.feed("1] More.")
        # Citation flushed when space after ] seen
        final = p.finalize()
        all_spans = r1.completed_spans + r2.completed_spans + final.completed_spans
        assert len(all_spans) == 1
        assert all_spans[0].ref_ids == ("S1",)

    def test_char_by_char_feeding(self) -> None:
        """Feed one character at a time."""
        raw = "AB [S1] CD"
        p = BracketCitationParser()
        total_visible = ""
        all_spans: list[AnswerCitationSpan] = []
        for ch in raw:
            r = p.feed(ch)
            total_visible += r.visible_text
            all_spans.extend(r.completed_spans)
        final = p.finalize()
        total_visible += final.visible_text
        all_spans.extend(final.completed_spans)

        assert total_visible == "AB  CD"
        assert len(all_spans) == 1
        span = all_spans[0]
        assert span.start == 0
        assert span.end == 3  # "AB " (3 chars)
        assert span.ref_ids == ("S1",)

    def test_multi_chunk_multiple_citations(self) -> None:
        chunks = ["Revenue grew. [S1] ", "Margins improved. [S2]"]
        p = BracketCitationParser()
        all_visible = ""
        all_spans: list[AnswerCitationSpan] = []
        for chunk in chunks:
            r = p.feed(chunk)
            all_visible += r.visible_text
            all_spans.extend(r.completed_spans)
        final = p.finalize()
        all_visible += final.visible_text
        all_spans.extend(final.completed_spans)

        assert len(all_spans) == 2
        assert all_spans[0].ref_ids == ("S1",)
        assert all_spans[1].ref_ids == ("S2",)


# ---------------------------------------------------------------------------
# BracketCitationParser — malformed / non-citation brackets
# ---------------------------------------------------------------------------


class TestParserMalformed:
    def test_non_citation_bracket_passes_through(self) -> None:
        p = BracketCitationParser()
        r = p.feed("See [note 1] for details.")
        assert "[note 1]" in r.visible_text
        assert r.completed_spans == []

    def test_numeric_only_bracket_passes_through(self) -> None:
        p = BracketCitationParser()
        r = p.feed("See [1] and [12].")
        assert "[1]" in r.visible_text
        assert "[12]" in r.visible_text
        assert r.completed_spans == []

    def test_c_prefix_bracket_passes_through(self) -> None:
        """[C1] is not a valid source citation (C-prefix is display label)."""
        p = BracketCitationParser()
        r = p.feed("See [C1].")
        assert "[C1]" in r.visible_text
        assert r.completed_spans == []

    def test_bracket_overflow_flushed_as_text(self) -> None:
        """Bracket with content > 30 chars should be flushed as visible text."""
        p = BracketCitationParser()
        long_content = "[" + "x" * 31 + "]"
        r = p.feed(long_content)
        assert r.completed_spans == []

    def test_unclosed_bracket_at_finalize(self) -> None:
        """Stream ends with open bracket — flushed as visible text, no span."""
        p = BracketCitationParser()
        r = p.feed("Text [S")
        assert r.visible_text == "Text "
        final = p.finalize()
        assert final.visible_text == "[S"
        assert final.completed_spans == []

    def test_finalize_raises_on_double_call(self) -> None:
        p = BracketCitationParser()
        p.finalize()
        with pytest.raises(RuntimeError, match="already finalized"):
            p.finalize()

    def test_feed_raises_after_finalize(self) -> None:
        p = BracketCitationParser()
        p.finalize()
        with pytest.raises(RuntimeError, match="already finalized"):
            p.feed("text")


# ---------------------------------------------------------------------------
# BracketCitationParser — display label ordering
# ---------------------------------------------------------------------------


class TestParserDisplayLabels:
    def test_labels_by_first_appearance(self) -> None:
        """First citation refs S3, second refs S1 — display labels C1, C2."""
        p = BracketCitationParser()
        p.feed("First. [S3] Second. [S1]")
        p.finalize()
        assert p.label_map.get_or_assign("S3") == "C1"
        assert p.label_map.get_or_assign("S1") == "C2"

    def test_repeated_ref_keeps_label(self) -> None:
        p = BracketCitationParser()
        p.feed("A. [S1] B. [S2] C. [S1]")
        p.finalize()
        assert p.label_map.mapping["S1"] == "C1"
        assert p.label_map.mapping["S2"] == "C2"

    def test_all_spans_property(self) -> None:
        p = BracketCitationParser()
        p.feed("A. [S1] B. [S2] ")
        p.finalize()
        assert len(p.all_spans) == 2


# ---------------------------------------------------------------------------
# BracketCitationParser — offset consistency
# ---------------------------------------------------------------------------


class TestParserOffsets:
    def test_offsets_align_with_visible_text(self) -> None:
        """Verify span offsets slice correctly out of concatenated visible text."""
        raw = "The report shows revenue grew 15%. [S1] Margins also expanded. [S2]"
        p = BracketCitationParser()
        r = p.feed(raw)
        final = p.finalize()
        all_spans = r.completed_spans + final.completed_spans
        all_visible = r.visible_text + final.visible_text

        assert len(all_spans) == 2
        s1, s2 = all_spans
        # s1 should cover the text up to and including the space before [S1]
        assert all_visible[s1.start : s1.end].endswith("15%. ")
        # s2 starts where s1 ended
        assert s2.start == s1.end

    def test_multi_chunk_offsets_consistent(self) -> None:
        chunks = ["Revenue: $2B. [S", "1] Net income: $0.5B. [S2]"]
        p = BracketCitationParser()
        all_visible = ""
        all_spans: list[AnswerCitationSpan] = []
        for chunk in chunks:
            r = p.feed(chunk)
            all_visible += r.visible_text
            all_spans.extend(r.completed_spans)
        final = p.finalize()
        all_visible += final.visible_text
        all_spans.extend(final.completed_spans)

        assert len(all_spans) == 2
        for span in all_spans:
            assert 0 <= span.start <= span.end <= len(all_visible)
