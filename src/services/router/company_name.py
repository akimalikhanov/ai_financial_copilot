"""The one company-name normalizer: stored as `documents.company_norm` at upload and
applied to the query side at resolution, so both sides always agree."""

from __future__ import annotations

import re
import unicodedata

_CORP_SUFFIXES = frozenset(
    {
        # English
        "limited",
        "ltd",
        "llc",
        "inc",
        "incorporated",
        "corp",
        "corporation",
        "company",
        "co",
        "plc",
        "lp",
        "llp",
        "pllc",
        "pc",
        "holdings",
        "holding",
        "group",
        "trust",
        "reit",
        # Australian/UK/SG/MY
        "pty",
        "pte",
        "sdn",
        "bhd",
        # German/Austrian/Swiss
        "gmbh",
        "ag",
        "kg",
        "kgaa",
        "se",
        # Dutch
        "bv",
        "nv",
        "cv",
        # French/Belgian
        "sa",
        "sas",
        "sarl",
        "sca",
        # Italian/Spanish/Portuguese
        "srl",
        "spa",
        "sl",
        # Nordic
        "ab",
        "oy",
        "oyj",
        "as",
        "asa",
        "aps",
        # Japanese romanized
        "kk",
        "gk",
    }
)

_BRACKETED = re.compile(r"\([^)]*\)|\[[^\]]*\]")
_NON_WORD = re.compile(r"[^\w\s]|_")


def normalize_company(name: str) -> str:
    """Lowercase, fold accents, drop bracketed text and punctuation, then strip trailing
    corporate suffixes until none is left. The last word is never stripped.

    "Blue Apron Holdings, Inc." -> "blue apron"
    "Microsoft Corporation (scanned 10p)" -> "microsoft"
    "Poste Italiane S.p.A." -> "poste italiane"
    """
    folded = unicodedata.normalize("NFKD", name)
    folded = "".join(c for c in folded if not unicodedata.combining(c)).lower()
    folded = _BRACKETED.sub(" ", folded)
    # Dots join abbreviations ("S.p.A." -> "spa"); other punctuation separates words.
    words = _NON_WORD.sub(" ", folded.replace(".", "")).split()
    while len(words) > 1 and words[-1] in _CORP_SUFFIXES:
        words.pop()
    return " ".join(words)
