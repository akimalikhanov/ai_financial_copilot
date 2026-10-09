"""The generated load-test fixtures must actually trip the detectors they exist to exercise.

A fixture that silently fails to trigger its path is worse than no fixture: the run still
produces numbers, they just measure something else. The brokenfont fixture already did this
once at 3 pages — ~1,950 extracted characters, just under text_quality's 2,000-character floor,
so `assess` declined to judge and the OCR fallback never fired.

Only the synthetic class is tested here. normal.pdf and scanned.pdf derive from a real filing
that is gitignored and absent in CI; classify_page_content is covered directly against
generated PDFs in test_text_quality.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from infra.loadtest.make_fixtures import build_brokenfont
from src.services.ingestion import text_quality

THRESHOLD = 0.02


@pytest.fixture(scope="module")
def brokenfont(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("fixtures") / "brokenfont.pdf"
    path.write_bytes(build_brokenfont())
    return path


def _extract(path: Path) -> str:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    try:
        return "".join(pdf[i].get_textpage().get_text_range() for i in range(len(pdf)))
    finally:
        pdf.close()


class TestBrokenFontFixture:
    def test_is_a_readable_pdf(self, brokenfont: Path) -> None:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(brokenfont))
        assert len(pdf) == 8
        pdf.close()

    def test_extracted_text_is_garbled(self, brokenfont: Path) -> None:
        garbled, ratio = text_quality.assess(_extract(brokenfont), threshold=THRESHOLD)
        assert garbled is True
        assert ratio < THRESHOLD

    def test_sample_clears_the_minimum_to_be_judged(self, brokenfont: Path) -> None:
        """The regression that matters: below text_quality._MIN_SAMPLE_CHARS the gate returns
        not-garbled regardless of how bad the text is, and the fixture does nothing."""
        assert len(_extract(brokenfont)) >= 2_000

    def test_has_a_text_layer_so_it_is_not_classified_as_a_scan(self, brokenfont: Path) -> None:
        """This fixture must reach the OCR converter via the garbled-text gate, not the
        scanned-page one — otherwise the two routes are not both covered."""
        verdict, _ = text_quality.classify_page_content(brokenfont)
        assert verdict == "text"

    def test_rendered_glyphs_are_not_themselves_shifted(self, brokenfont: Path) -> None:
        """The point of the wrong-ToUnicode approach: the page still *draws* readable English,
        so OCR recovers real text. Writing pre-shifted characters would render nonsense and
        OCR would return the same nonsense, exercising the failure branch instead."""
        content = brokenfont.read_bytes()
        assert b"(ACME CORPORATION ANNUAL REPORT 2024) Tj" in content
