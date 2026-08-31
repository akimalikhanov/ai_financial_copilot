"""Unit tests for broken-font-encoding detection (Phase 8)."""

from __future__ import annotations

from src.services.ingestion import text_quality

THRESHOLD = 0.02

# Real prose from an annual report.
CLEAN = (
    "Conflicts of interest may arise because certain of our directors and officers are also "
    "directors of MGM China, the holding company for MGM Grand Paradise, which owns and "
    "operates MGM Macau and MGM Cotai. As a result of the initial public offering of shares "
    "of MGM China common stock, we are subject to additional regulatory oversight in Macau. "
    "The board of directors has determined that these arrangements are in the best interests "
    "of the company and of its shareholders as a whole, and has approved them accordingly. "
) * 6

# The same passage as it actually came out of Docling: glyph codes offset from ASCII.
SHIFTED = (
    "$V D UHVXOW RI WKH LQLWLDO SXEOLF RIIHULQJ RI VKDUHV RI 0*0 &KLQD FRPPRQ VWRFN LQ "
    "FRQGLWLRQ UHVXOWV RI RSHUDWLRQV DQG FDVK IORZV ,Q DGGLWLRQ WKH RXWEUHDN RI "
    "LQIHFWLRXV GLVHDVHV VXFK DV &29,' PD\\ VHYHUHO\\ GLVUXSW GRPHVWLF WUDYHO "
) * 12

# Codes with no mapping at all: ~40 characters of boilerplate per source character.
GLYPH_SPAM = (
    "GLYPH<c=3,font=/LBKBHL+TimesNewRoman>GLYPH<c=22,font=/LBKBHL+TimesNewRoman>"
    "GLYPH<c=17,font=/LBKBHL+TimesNewRoman>"
) * 40


class TestStopwordRatio:
    def test_real_prose_scores_well_above_threshold(self) -> None:
        assert text_quality.stopword_ratio(CLEAN) > 0.06

    def test_shifted_text_scores_near_zero(self) -> None:
        assert text_quality.stopword_ratio(SHIFTED) < 0.005

    def test_empty_text_is_zero_not_an_error(self) -> None:
        assert text_quality.stopword_ratio("") == 0.0


class TestAssess:
    def test_clean_prose_passes(self) -> None:
        garbled, _ = text_quality.assess(CLEAN, threshold=THRESHOLD)
        assert garbled is False

    def test_shifted_text_is_caught(self) -> None:
        garbled, ratio = text_quality.assess(SHIFTED, threshold=THRESHOLD)
        assert garbled is True
        assert ratio < THRESHOLD

    def test_glyph_markers_are_caught_regardless_of_sample_size(self) -> None:
        """The unmapped-glyph shape is unambiguous, so it does not wait for a big sample."""
        garbled, _ = text_quality.assess(GLYPH_SPAM[:500], threshold=THRESHOLD)
        assert garbled is True

    def test_html_escaped_glyph_markers_are_caught(self) -> None:
        """Docling escapes '<' in table cells, so the marker also appears as 'GLYPH&lt;'."""
        garbled, _ = text_quality.assess(
            GLYPH_SPAM.replace("<", "&lt;").replace(">", "&gt;")[:500], threshold=THRESHOLD
        )
        assert garbled is True

    def test_short_sample_is_not_judged(self) -> None:
        """A table-only document has little prose to score and must not be failed for it."""
        garbled, _ = text_quality.assess("2024 2023 1,204 998 3,551", threshold=THRESHOLD)
        assert garbled is False

    def test_predominantly_shifted_document_is_caught(self) -> None:
        """The real document interleaved decoded and mis-decoded runs, garbage dominating."""
        garbled, _ = text_quality.assess(CLEAN + SHIFTED * 30, threshold=THRESHOLD)
        assert garbled is True

    def test_partly_garbled_document_is_missed_by_the_ratio_alone(self) -> None:
        """Known limit: the ratio is a document-level average, so a document that is only
        part garbled averages back above the threshold and is not caught by it."""
        garbled, ratio = text_quality.assess(CLEAN + SHIFTED * 4, threshold=THRESHOLD)
        assert garbled is False
        assert ratio > THRESHOLD

    def test_partly_garbled_document_is_still_caught_via_glyph_markers(self) -> None:
        """Which is why the glyph signal exists: it is a density check, not an average, so it
        fires on the same document as soon as any unmapped-glyph runs are present."""
        garbled, _ = text_quality.assess(CLEAN + SHIFTED * 4 + GLYPH_SPAM, threshold=THRESHOLD)
        assert garbled is True


class TestSampleDocumentText:
    def test_reads_text_items_and_respects_budget(self) -> None:
        class _Item:
            def __init__(self, text: str) -> None:
                self.text = text

        class _Doc:
            texts = [_Item("hello world") for _ in range(100)]

        sample = text_quality.sample_document_text(_Doc(), budget=50)  # pyright: ignore[reportArgumentType]
        assert 50 <= len(sample) < 100
        assert sample.startswith("hello world")

    def test_skips_items_without_text(self) -> None:
        class _Doc:
            texts = [object(), object()]

        assert text_quality.sample_document_text(_Doc()) == ""  # pyright: ignore[reportArgumentType]
