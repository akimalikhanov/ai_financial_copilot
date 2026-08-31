"""Detects text that parsed into garbage because the PDF's font encoding is broken.

A PDF whose embedded subset fonts carry a missing or wrong ToUnicode CMap yields a text layer
of raw glyph codes rather than characters. Two shapes show up, often in the same document:

    "ILQDQFLDO"                                    -> letters offset from ASCII ("financial")
    "GLYPH<c=3,font=/LBKBHL+TimesNewRoman>"        -> codes with no mapping at all

The second is also a token bomb: ~40 characters of boilerplate per source character, which is
how one 120-page filing produced 2.58M tokens and 4,034 chunks instead of ~250.

Neither shape fails the parse — Docling reports SUCCESS — so nothing downstream notices until
the text has been embedded and indexed as retrievable evidence.
"""

from __future__ import annotations

import re
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
