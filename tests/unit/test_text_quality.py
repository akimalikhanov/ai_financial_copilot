"""Unit tests for broken-font-encoding detection (Phase 8)."""

from __future__ import annotations

from pathlib import Path

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


# --- Scanned-vs-blank classification -------------------------------------------------------
#
# Built as real PDFs rather than stubs: the whole point of this classifier is that it reads
# PDF page objects, and the bug it was written to fix was a stubbed model of Docling's output
# that did not match what Docling actually produces for a scan (texts=0, pictures=0).

DPI = 100
PAGE_PX = (int(8.5 * DPI), int(11 * DPI))

PROSE_LINES = [
    "ACME CORPORATION ANNUAL REPORT",
    "Total revenue for the year was $4,821 million, an increase of",
    "12.4% over the prior year, driven by growth in services and",
    "continued expansion of the subscription base. Operating margin",
    "improved to 18.2% from 16.7%, reflecting operating leverage.",
    "Cash and cash equivalents totaled $1,204 million at year end.",
]


def _image_pdf(path: Path, pages: int, *, ink: bool) -> Path:
    """A PDF whose pages are images — what a scanner or a screenshot produces.

    `ink=False` is the control: image-backed but carrying nothing, the case OCR must not be
    spent on.
    """
    from PIL import Image, ImageDraw

    imgs = []
    for _ in range(pages):
        img = Image.new("RGB", PAGE_PX, "white")
        if ink:
            draw = ImageDraw.Draw(img)
            y = 120
            for line in PROSE_LINES:
                draw.text((100, y), line, fill=(10, 10, 10))
                y += 40
        imgs.append(img)
    imgs[0].save(path, save_all=True, append_images=imgs[1:], resolution=float(DPI))
    return path


def _text_pdf(path: Path, pages: int, *, lines_per_page: int = 8) -> Path:
    """A born-digital PDF: real text-drawing operators, no page image.

    Hand-built rather than written with pypdfium2, which can read text objects but not
    create them. A PDF is plain bytes with a computed xref table, so this stays exact and
    dependency-free.
    """
    objs: list[bytes] = []
    font_num = 3 + pages * 2
    kids = " ".join(f"{3 + i * 2} 0 R" for i in range(pages))
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {pages} >>".encode())
    for i in range(pages):
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {4 + i * 2} 0 R "
            f"/Resources << /Font << /F1 {font_num} 0 R >> >> >>".encode()
        )
        y = 700
        lines = []
        for n in range(lines_per_page):
            text = PROSE_LINES[n % len(PROSE_LINES)].replace("$", r"\$")
            lines.append(f"BT /F1 11 Tf 60 {y} Td ({text}) Tj ET")
            y -= 30
        stream = "\n".join(lines).encode()
        objs.append(
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
        )
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for num, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{num} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode() + b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n".encode()
    )
    path.write_bytes(bytes(out))
    return path


class TestClassifyPageContent:
    def test_born_digital_pdf_is_text(self, tmp_path: Path) -> None:
        verdict, stats = text_quality.classify_page_content(_text_pdf(tmp_path / "d.pdf", 3))
        assert verdict == "text"
        assert stats.text_pages == 3
        assert stats.scanned_pages == 0

    def test_scanned_pdf_is_detected(self, tmp_path: Path) -> None:
        """The case the parsed DoclingDocument cannot see: with DOCLING_DO_OCR=false this
        document yields texts=0, pictures=0, tables=0 — identical to a blank page."""
        verdict, stats = text_quality.classify_page_content(
            _image_pdf(tmp_path / "s.pdf", 3, ink=True)
        )
        assert verdict == "scanned"
        assert stats.scanned_pages == 3
        assert stats.scanned_fraction == 1.0

    def test_blank_image_pdf_is_still_scanned_shaped(self, tmp_path: Path) -> None:
        """A scan of blank paper is image-backed and textless, so it reads as scanned. OCR
        will find nothing and the parse lands on scanned_no_text, which is the honest
        outcome — the classifier cannot tell blank paper from paper it has not read yet."""
        verdict, _ = text_quality.classify_page_content(
            _image_pdf(tmp_path / "b.pdf", 2, ink=False)
        )
        assert verdict == "scanned"

    def test_empty_pdf_with_no_content_is_blank(self, tmp_path: Path) -> None:
        """No text operators and no image at all: OCR would only spend ~4.3s a page
        confirming the document is still empty."""
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument.new()
        for _ in range(3):
            pdf.new_page(612, 792)
        pdf.save(str(tmp_path / "e.pdf"))

        verdict, stats = text_quality.classify_page_content(tmp_path / "e.pdf")
        assert verdict == "blank"
        assert stats.blank_pages == 3
        assert stats.considered_pages == 0

    def test_mostly_digital_with_scanned_exhibits_stays_text(self, tmp_path: Path) -> None:
        import pypdfium2 as pdfium

        digital = _text_pdf(tmp_path / "a.pdf", 8)
        scan = _image_pdf(tmp_path / "c.pdf", 2, ink=True)
        merged = pdfium.PdfDocument.new()
        merged.import_pages(pdfium.PdfDocument(str(digital)))
        merged.import_pages(pdfium.PdfDocument(str(scan)))
        merged.save(str(tmp_path / "m.pdf"))

        verdict, stats = text_quality.classify_page_content(tmp_path / "m.pdf")
        assert verdict == "text"
        assert stats.text_pages == 8
        assert stats.scanned_pages == 2

    def test_majority_scanned_document_qualifies(self, tmp_path: Path) -> None:
        import pypdfium2 as pdfium

        digital = _text_pdf(tmp_path / "a.pdf", 2)
        scan = _image_pdf(tmp_path / "c.pdf", 8, ink=True)
        merged = pdfium.PdfDocument.new()
        merged.import_pages(pdfium.PdfDocument(str(digital)))
        merged.import_pages(pdfium.PdfDocument(str(scan)))
        merged.save(str(tmp_path / "m.pdf"))

        verdict, stats = text_quality.classify_page_content(tmp_path / "m.pdf")
        assert verdict == "scanned"
        assert stats.scanned_fraction == 0.8

    def test_unreadable_pdf_does_not_raise(self, tmp_path: Path) -> None:
        """A parse that already succeeded must never be failed by this check."""
        bad = tmp_path / "bad.pdf"
        bad.write_bytes(b"%PDF-1.4\nnot actually a pdf")
        verdict, _ = text_quality.classify_page_content(bad)
        assert verdict == "text"

    def test_missing_file_does_not_raise(self, tmp_path: Path) -> None:
        verdict, _ = text_quality.classify_page_content(tmp_path / "nope.pdf")
        assert verdict == "text"


class TestLooksTextless:
    def test_normal_parse_is_not_probed(self) -> None:
        class _Item:
            text = CLEAN

        class _Doc:
            texts = [_Item()]

        assert text_quality.looks_textless(_Doc(), 1) is False  # pyright: ignore[reportArgumentType]

    def test_empty_parse_is_probed(self) -> None:
        class _Doc:
            texts = []

        assert text_quality.looks_textless(_Doc(), 3) is True  # pyright: ignore[reportArgumentType]

    def test_stray_header_on_a_long_scan_is_probed(self) -> None:
        """A Bates stamp on each page is far below the per-page floor."""

        class _Item:
            text = "ACME 10-K p.1"

        class _Doc:
            texts = [_Item() for _ in range(20)]

        assert text_quality.looks_textless(_Doc(), 20) is True  # pyright: ignore[reportArgumentType]

    def test_zero_pages_is_not_probed(self) -> None:
        class _Doc:
            texts = []

        assert text_quality.looks_textless(_Doc(), 0) is False  # pyright: ignore[reportArgumentType]
