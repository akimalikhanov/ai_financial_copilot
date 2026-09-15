"""Detects PDFs whose text layer did not survive parsing, in two unrelated ways.

**Garbled text.** A PDF whose embedded subset fonts carry a missing or wrong ToUnicode CMap
yields a text layer of raw glyph codes rather than characters. Two shapes show up, often in
the same document:

    "ILQDQFLDO"                                    -> letters offset from ASCII ("financial")
    "GLYPH<c=3,font=/LBKBHL+TimesNewRoman>"        -> codes with no mapping at all

The second is also a token bomb: ~40 characters of boilerplate per source character, which is
how one 120-page filing produced 2.58M tokens and 4,034 chunks instead of ~250.

**No text layer at all.** A scanned or photocopied PDF is page images with nothing to extract.
With DOCLING_DO_OCR=false (the lean parse config) Docling emits a near-empty document and still
reports SUCCESS, so the document completes with zero chunks and is marked ready — indexed,
searchable, and empty. `classify_page_content` separates that from a genuinely blank document,
because the two want opposite handling: one is worth paying full-page OCR for, the other is a
document with nothing in it and OCR would only burn GPU to confirm it.

Neither shape fails the parse — Docling reports SUCCESS in both cases — so nothing downstream
notices until the text has been embedded and indexed as retrievable evidence.
"""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument

# Common English function words. Deliberately tiny: any encoding that shifts or drops character
# codes destroys all of them at once, so a handful discriminates as well as a dictionary would.
_STOPWORDS = re.compile(r"\b(?:the|and|of|to|in|for|is|that|with|as)\b", re.IGNORECASE)
_WORDS = re.compile(r"\b\w+\b")
_GLYPH_MARKER = re.compile(r"GLYPH(?:<|&lt;)")

# Characters of document text to score. Enough to be representative, small enough to stay cheap
# on a 4,000-chunk document.
_SAMPLE_BUDGET = 50_000
# Below this, the sample is too small for the ratio to mean anything — a document that is almost
# entirely tables has little prose to score, and must not be failed for it.
_MIN_SAMPLE_CHARS = 2_000
# One marker per this many characters is already far past anything legitimate text produces.
_GLYPH_DENSITY_LIMIT = 0.001


def stopword_ratio(text: str) -> float:
    """Fraction of words that are common English function words.

    Real prose sits at 0.06 and up; text from a broken font encoding sits near zero, because
    every stopword is mangled too. Measured across a 50-document corpus the gap was 200x.
    """
    words = _WORDS.findall(text)
    if not words:
        return 0.0
    return len(_STOPWORDS.findall(text)) / len(words)


def glyph_marker_density(text: str) -> float:
    """Unmapped-glyph markers per character. Unambiguous when present, but absent on pages whose
    codes all shift cleanly, so it cannot be the only signal."""
    if not text:
        return 0.0
    return len(_GLYPH_MARKER.findall(text)) / len(text)


def sample_document_text(document: DoclingDocument, budget: int = _SAMPLE_BUDGET) -> str:
    """Concatenate the document's text items up to a character budget.

    Reads `document.texts` rather than exporting Markdown: the export is re-done later in the
    pipeline anyway, and this avoids paying for it twice on exactly the oversized documents
    this check exists to catch.
    """
    parts: list[str] = []
    total = 0
    for item in document.texts:
        text = getattr(item, "text", None)
        if not text:
            continue
        parts.append(text)
        total += len(text)
        if total >= budget:
            break
    return "\n".join(parts)


# --- Scanned-vs-blank classification -------------------------------------------------------
_SCAN_TEXT_OBJECTS_PER_PAGE = 5
# Fraction of page area covered by image content for the page to be a page image rather than
# a figure. Secondary to the operator count and never sufficient alone: a real filing's cover
# page measured coverage 1.029 (full-bleed background) while carrying 18 text operators.
_SCAN_IMAGE_COVERAGE = 0.5
# Fraction of *considered* pages that must look scanned before the document is called scanned.
# Below 1.0 so a born-digital filing with a few scanned exhibits appended still qualifies —
# those exhibits are exactly the pages a reader wants and the only ones OCR would recover.
_SCAN_PAGE_FRACTION = 0.5
# Extracted characters per page below which the parse is suspicious enough to be worth
# probing the PDF. Keeps the probe (~2ms/page) off the normal path entirely.
_PROBE_CHARS_PER_PAGE = 50


@dataclass(frozen=True)
class PageContentStats:
    """Per-document page accounting behind the scanned/blank/text verdict."""

    total_pages: int
    considered_pages: int  # pages with neither text nor image content excluded
    scanned_pages: int  # no meaningful text, but image content covering most of the page
    blank_pages: int  # neither text nor image content
    text_pages: int

    @property
    def scanned_fraction(self) -> float:
        if self.considered_pages == 0:
            return 0.0
        return self.scanned_pages / self.considered_pages


def looks_textless(document: DoclingDocument, page_count: int) -> bool:
    """Cheap precondition: did the parse extract so little text that the PDF is worth probing?

    Guards the probe so it never runs on a document that parsed normally. Deliberately coarse
    — a false positive here costs ~2ms a page and is then corrected by the probe itself.
    """
    if page_count <= 0:
        return False
    total = sum(len(t) for item in document.texts if (t := getattr(item, "text", None)))
    return (total / page_count) < _PROBE_CHARS_PER_PAGE


def _page_signature(page, page_area: float) -> tuple[int, float]:
    """(text-drawing operator count, image area) for one pdfium page.

    Stops counting operators once past the threshold: a 200-page born-digital document has
    hundreds per page and the exact count is never used, only the comparison.
    """
    import pypdfium2.raw as pdfium_c

    text_objects = 0
    image_area = 0.0
    for obj in page.get_objects():
        obj_type = getattr(obj, "type", None)
        if obj_type == pdfium_c.FPDF_PAGEOBJ_TEXT:
            text_objects += 1
            if text_objects > _SCAN_TEXT_OBJECTS_PER_PAGE:
                # Unambiguously a text page; the image tally cannot change that verdict.
                return text_objects, image_area
        elif obj_type == pdfium_c.FPDF_PAGEOBJ_IMAGE:
            try:
                left, bottom, right, top = obj.get_bounds()
            except Exception:
                continue
            covered = abs((right - left) * (top - bottom))
            # Clamp per object: a full-bleed image plus overlapping crops must not sum to
            # several times the page and turn a text page into a "scan".
            image_area += min(covered, page_area) if page_area else covered
    return text_objects, image_area


def classify_page_content(pdf_path: Path) -> tuple[str, PageContentStats]:
    """Return (verdict, stats) where verdict is "text" | "scanned" | "blank".

    Reads the PDF's own page objects, not the parsed DoclingDocument. That is not a stylistic
    choice: with DOCLING_DO_OCR=false a scanned page parses to texts=0, pictures=0, tables=0,
    identical to a blank page, so the parsed document carries no signal to classify on.

    A page is "scanned" when it draws almost no text and most of its area is image. The
    operator count leads and the coverage only confirms — a real filing's cover page measured
    full-page image coverage while carrying 18 text operators, so coverage alone would send
    well-formed documents for a needless OCR re-parse.

    "blank" is reported rather than folded into "scanned" because the two want opposite
    handling: OCR recovers a scan, and only wastes ~4.3s a page confirming that an empty
    document is still empty.

    Never raises: an unreadable PDF returns "text" (do nothing) rather than failing a parse
    that has already succeeded.
    """
    stats = PageContentStats(0, 0, 0, 0, 0)
    try:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(pdf_path))
    except Exception:
        return "text", stats

    scanned = blank = text_pages = 0
    total_pages = 0
    try:
        total_pages = len(pdf)
        for index in range(total_pages):
            page = pdf[index]
            width, height = page.get_width(), page.get_height()
            page_area = float(width) * float(height)
            text_objects, image_area = _page_signature(page, page_area)
            coverage = (image_area / page_area) if page_area > 0 else 0.0

            if text_objects > _SCAN_TEXT_OBJECTS_PER_PAGE:
                text_pages += 1
            elif coverage >= _SCAN_IMAGE_COVERAGE:
                scanned += 1
            else:
                blank += 1
    except Exception:
        # Partial read: trust what was counted only if it already proves the document has
        # text. Otherwise say nothing rather than sending a readable PDF for OCR.
        if text_pages == 0:
            return "text", stats
    finally:
        with contextlib.suppress(Exception):
            pdf.close()

    considered = scanned + text_pages
    stats = PageContentStats(
        total_pages=total_pages,
        considered_pages=considered,
        scanned_pages=scanned,
        blank_pages=blank,
        text_pages=text_pages,
    )

    if considered == 0:
        return ("blank" if total_pages else "text"), stats
    if stats.scanned_fraction >= _SCAN_PAGE_FRACTION:
        return "scanned", stats
    return "text", stats


def assess(text: str, *, threshold: float) -> tuple[bool, float]:
    """Return (is_garbled, stopword_ratio) for a text sample.

    Garbled when unmapped-glyph markers are dense, or when there is enough prose to judge and
    it contains almost no function words.
    """
    ratio = stopword_ratio(text)
    if glyph_marker_density(text) > _GLYPH_DENSITY_LIMIT:
        return True, ratio
    if len(text) < _MIN_SAMPLE_CHARS:
        return False, ratio
    return ratio < threshold, ratio
