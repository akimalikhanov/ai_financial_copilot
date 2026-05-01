"""Invisible Unicode codepoint ranges and homoglyph fold map."""

from __future__ import annotations

# Ranges of codepoints considered invisible / structurally zero-width.
# Each entry is (start, end) inclusive.
INVISIBLE_RANGES: list[tuple[int, int]] = [
    (0x00AD, 0x00AD),  # soft hyphen
    (0x180E, 0x180E),  # Mongolian vowel separator
    (0x200B, 0x200F),  # zero-width space / non-joiner / joiner / LRM / RLM
    (0x202A, 0x202E),  # LRE / RLE / PDF / LRO / RLO (bidi embedding)
    (0x2060, 0x206F),  # word joiner, invisible separators, inhibit-swap, etc.
    (0x3164, 0x3164),  # Hangul filler
    (0xFE00, 0xFE0F),  # variation selectors 1-16
    (0xFEFF, 0xFEFF),  # BOM / zero-width no-break space (when not at position 0)
    (0xFFA0, 0xFFA0),  # halfwidth Hangul filler
    (0xE0000, 0xE007F),  # tag block base
    (0xE0100, 0xE01EF),  # variation selectors supplement
]


def _build_invisible_set() -> frozenset[int]:
    result: set[int] = set()
    for start, end in INVISIBLE_RANGES:
        result.update(range(start, end + 1))
    return frozenset(result)


INVISIBLE_CODEPOINTS: frozenset[int] = _build_invisible_set()


# High-frequency Cyrillic/Greek → Latin homoglyph fold map.
# Keys are the confusable character, values are their ASCII lookalikes.
HOMOGLYPH_MAP: dict[str, str] = {
    # Cyrillic
    "а": "a",  # а → a
    "е": "e",  # е → e
    "о": "o",  # о → o
    "р": "p",  # р → p
    "с": "c",  # с → c
    "х": "x",  # х → x
    "і": "i",  # і → i (Ukrainian i)
    "у": "y",  # у → y
    "в": "v",  # в → v
    "А": "A",  # А → A
    "В": "B",  # В → B
    "Е": "E",  # Е → E
    "К": "K",  # К → K
    "М": "M",  # М → M
    "Н": "H",  # Н → H
    "О": "O",  # О → O
    "Р": "P",  # Р → P
    "С": "C",  # С → C
    "Т": "T",  # Т → T
    "Х": "X",  # Х → X
    # Greek
    "α": "a",  # α → a
    "ε": "e",  # ε → e
    "ο": "o",  # ο → o
    "ρ": "p",  # ρ → p
    "ν": "v",  # ν → v
    "Α": "A",  # Α → A
    "Β": "B",  # Β → B
    "Ε": "E",  # Ε → E
    "Κ": "K",  # Κ → K
    "Μ": "M",  # Μ → M
    "Ν": "N",  # Ν → N
    "Ο": "O",  # Ο → O
    "Ρ": "P",  # Ρ → P
    "Τ": "T",  # Τ → T
    "Χ": "X",  # Χ → X
    "Υ": "Y",  # Υ → Y
    "Ζ": "Z",  # Ζ → Z
}
