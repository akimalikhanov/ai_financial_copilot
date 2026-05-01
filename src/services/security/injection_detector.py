"""Prompt injection detection — Layer A (pattern-based, no LLM).

Pure module: no I/O, no side effects, fully synchronous.
All public functions are thread-safe (compiled regex objects are immutable).
"""

from __future__ import annotations

import base64
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Literal

from src.services.security.unicode_tables import HOMOGLYPH_MAP, INVISIBLE_CODEPOINTS

# ---------------------------------------------------------------------------
# Pre-compiled regex patterns
# ---------------------------------------------------------------------------

# Role-marker tokens used by common chat-ML formats.
_RE_ROLE_MARKER = re.compile(
    r"<\|(?:im_start|im_end|endoftext|start_header_id|end_header_id|eot_id|system|user|assistant)\|>"
    r"|<<SYS>>|<</SYS>>|<start_of_turn>|<end_of_turn>|\[/?INST\]"
    r"|(?m:^(?:Human|Assistant|H|A):\s)"
    r"|(?m:^###\s+(?:Instruction|Response|System):)",
    re.IGNORECASE,
)

# "Ignore / disregard / forget … previous / prior / all … instructions"
_RE_OVERRIDE = re.compile(
    r"""
    \b(?:ignore|disregard|forget|override)\b
    (?:\s+\w+){0,3}\s+
    \b(?:previous|prior|above|earlier|all|the\s+(?:system|original))\s+
    \b(?:instructions?|prompts?|rules?|directives?|guidelines?|context)\b
    """,
    re.VERBOSE | re.IGNORECASE,
)

# "Reveal / print / output … your system prompt / original instructions"
# Strong nouns (system prompt, initial/original instructions) score alone.
# Weak nouns (rules, guidelines, configuration) only score when another rule
# has already fired — they're too common in financial questions to score alone.
_RE_PROBE_STRONG = re.compile(
    r"""
    \b(?:reveal|print|output|show|display|repeat|echo|dump|tell\s+me|what\s+(?:are|is|were))\b
    (?:\s+\w+){0,5}\s+
    \b(?:your\s+)?(?:system\s+prompt|initial\s+instructions?|original\s+prompt)\b
    """,
    re.VERBOSE | re.IGNORECASE,
)
_RE_PROBE_WEAK = re.compile(
    r"""
    \b(?:reveal|print|output|show|display|repeat|echo|dump|tell\s+me|what\s+(?:are|is|were))\b
    (?:\s+\w+){0,5}\s+
    \b(?:(?:hidden|internal|secret|system)\s+)?(?:instructions?|rules?|guidelines?|configuration|directives?)\b
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Markdown / ASCII-art block headers announcing new instructions.
# Kept narrow to avoid FPs on common financial document markup:
#   - [Priority] / [Important] board memos → require only the attack-specific set
#   - === System Overview === → require instruction/prompt noun on same line
#   - >>> New market data → require instruction/prompt noun on same line
#   - <system>Note: ...</system> in iXBRL → only match exact role-injection tag forms
_RE_INSTRUCTION_BLOCK = re.compile(
    r"""
    (?im)
    ^(?:
        \#{2,}\s*(?:new|updated|important|priority)\s+(?:instructions?|prompts?|directives?)
      | -{3,}\s*(?:new|updated|important)\s+(?:instructions?|prompts?)
      | ={3,}\s*(?:new|system|important)\s+(?:instructions?|prompts?|directives?)
      | >{2,}\s*(?:new|system|admin)\s+(?:instructions?|prompts?|directives?)
      | \[(?:system|admin|root)\]
      | </?(?:admin|instructions?)>
      | begin\s+(?:new\s+)?(?:instructions?|prompts?|directives?)
    )
    """,
    re.VERBOSE,
)

# Unambiguously adversarial role patterns — score unconditionally.
# "you are now", "pretend to be", "roleplay as", and "act as <AI-role>" are
# unambiguous enough to stand alone without requiring another rule to fire.
_RE_ROLE_REASSIGN_STRONG = re.compile(
    r"""
    \byou\s+are\s+(?:now|going\s+to\s+be)\b
    | \bpretend\s+(?:to\s+be|you(?:'re|\s+are))\b
    | \broleplay\s+as\b
    | \bact\s+as\s+(?:a\s+)?(?:different\s+)?(?:ai|assistant|bot|gpt|llm|model|persona|character|admin|root|sudo|system|unrestricted|jailbreak)\b
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Weak role pattern — only scores when another rule has already fired.
# "act as" alone is too common in legal/financial prose ("act as agent",
# "act as collateral") to score unconditionally.
_RE_ROLE_REASSIGN_WEAK = re.compile(r"\bact\s+as\b", re.IGNORECASE)

# Base64 blob: ≥60 chars of standard or URL-safe alphabet, optional padding.
_RE_BASE64 = re.compile(r"[A-Za-z0-9+/\-_]{60,}={0,2}")

# Hex blob: ≥80 hex chars as a word.
_RE_HEX = re.compile(r"\b[0-9a-fA-F]{80,}\b")

# Instructional-density keywords (for chunk scanning).
_RE_INSTRUCTIONAL = re.compile(
    r"\b(?:you\s+must|do\s+not|please|output|respond\s+with|your\s+task|new\s+task)\b",
    re.IGNORECASE,
)

# Absolute keyword count threshold for instructional_density.
# At 700-900 token chunks, even 10 keywords gives ratio ~0.012 — a ratio-based
# threshold of 0.15 would never fire. Instead, ≥8 distinct keyword hits signals
# deliberate stuffing regardless of chunk length.
_INSTRUCTIONAL_DENSITY_MIN_KEYWORDS = 8


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass
class InjectionSignal:
    score: int
    severity: Literal["block", "flag", "clean"]
    matched_rules: list[str] = field(default_factory=list)
    stripped_chars: int = 0
    sanitized_text: str = ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _strip_invisible(text: str) -> tuple[str, int]:
    """Remove invisible/control Unicode codepoints. Returns (cleaned, count_removed)."""
    out: list[str] = []
    removed = 0
    for ch in text:
        if ord(ch) in INVISIBLE_CODEPOINTS:
            removed += 1
        else:
            out.append(ch)
    return "".join(out), removed


def _fold_homoglyphs(text: str) -> str:
    """Replace high-frequency confusable characters with their ASCII equivalents."""
    if not any(ch in HOMOGLYPH_MAP for ch in text):
        return text
    return "".join(HOMOGLYPH_MAP.get(ch, ch) for ch in text)


def _normalize(text: str) -> tuple[str, int]:
    """
    Full normalization pipeline:
      1. NFKC (defeats fullwidth / mathematical-alphanumeric evasion)
      2. Strip invisible Unicode
      3. Homoglyph fold

    Returns (sanitized_text, stripped_chars).
    """
    normalized = unicodedata.normalize("NFKC", text)
    stripped, count = _strip_invisible(normalized)
    folded = _fold_homoglyphs(stripped)
    return folded, count


def _try_decode_base64(blob: str) -> str | None:
    """Return decoded UTF-8 text from a base64 blob, or None if decoding fails.

    Accepts both standard (+/) and URL-safe (-_) alphabets. Uses validate=False
    because the regex already enforced the character set; strict validation would
    reject URL-safe chars when passed to the standard decoder.
    """
    # Normalise URL-safe alphabet to standard before decoding.
    normalised = blob.replace("-", "+").replace("_", "/")
    padded = normalised + "=" * (-len(normalised) % 4)
    try:
        decoded_bytes = base64.b64decode(padded, validate=False)
        return decoded_bytes.decode("utf-8", errors="strict")
    except Exception:
        return None


def _try_decode_hex(blob: str) -> str | None:
    """Return decoded UTF-8 text from a hex blob, or None if decoding fails."""
    try:
        decoded_bytes = bytes.fromhex(blob)
        return decoded_bytes.decode("utf-8", errors="strict")
    except Exception:
        return None


def _score_patterns(text: str, *, include_density: bool = False) -> tuple[int, list[str]]:
    """
    Apply pattern rules to already-normalized text.
    Returns (cumulative_score, matched_rule_names).
    Does NOT handle invisible stripping — that's done in _normalize.
    """
    score = 0
    rules: list[str] = []

    # role_marker_token — strip side-effect is handled at call site.
    if _RE_ROLE_MARKER.search(text):
        score += 2
        rules.append("role_marker_token")

    # override_instruction
    if _RE_OVERRIDE.search(text):
        score += 2
        rules.append("override_instruction")

    # system_prompt_probe — strong nouns score alone; weak nouns only when
    # another rule already fired (too common in legitimate financial questions).
    if _RE_PROBE_STRONG.search(text) or _RE_PROBE_WEAK.search(text) and rules:
        score += 2
        rules.append("system_prompt_probe")

    # instruction_block_marker
    if _RE_INSTRUCTION_BLOCK.search(text):
        score += 2
        rules.append("instruction_block_marker")

    # role_reassignment — strong patterns score +2 unconditionally (unambiguous
    # attacks like "act as AI", "pretend to be", "roleplay as"); weak "act as"
    # scores +1 only when another rule has already fired.
    if _RE_ROLE_REASSIGN_STRONG.search(text):
        score += 2
        rules.append("role_reassignment")
    elif _RE_ROLE_REASSIGN_WEAK.search(text) and rules:
        score += 1
        rules.append("role_reassignment")

    # base64_blob — decode and re-run rules recursively (one level deep).
    for m in _RE_BASE64.finditer(text):
        decoded = _try_decode_base64(m.group())
        if decoded:
            sub_score, sub_rules = _score_patterns(decoded)
            if sub_score > 0:
                score += sub_score
                rules.append("base64_blob")
                rules.extend(sub_rules)
                break  # one match is enough to count the blob rule

    # hex_blob — same approach.
    for m in _RE_HEX.finditer(text):
        decoded = _try_decode_hex(m.group())
        if decoded:
            sub_score, sub_rules = _score_patterns(decoded)
            if sub_score > 0:
                score += sub_score
                rules.append("hex_blob")
                rules.extend(sub_rules)
                break

    # instructional_density (chunks only).
    if include_density:
        keyword_count = len(_RE_INSTRUCTIONAL.findall(text))
        if keyword_count >= _INSTRUCTIONAL_DENSITY_MIN_KEYWORDS:
            score += 1
            rules.append("instructional_density")

    return score, rules


def _strip_role_markers(text: str) -> str:
    """Remove role-marker tokens from text (unconditional sanitization)."""
    return _RE_ROLE_MARKER.sub("", text)


def _severity_user(score: int) -> Literal["block", "flag", "clean"]:
    if score >= 2:
        return "block"
    if score == 1:
        return "flag"
    return "clean"


def _severity_chunk(score: int) -> Literal["block", "flag", "clean"]:
    if score >= 3:
        return "block"
    if score >= 1:
        return "flag"
    return "clean"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scan_user_input(text: str) -> InjectionSignal:
    """Scan a raw user query for prompt injection signals.

    Thresholds: score ≥ 2 → block, score == 1 → flag, score == 0 → clean.
    `sanitized_text` always has invisibles and role markers stripped.
    """
    sanitized, stripped = _normalize(text)
    sanitized = _strip_role_markers(sanitized)

    score, rules = _score_patterns(sanitized, include_density=False)

    if stripped > 0:
        score += 1
        rules.insert(0, "invisible_unicode")

    severity = _severity_user(score)

    return InjectionSignal(
        score=score,
        severity=severity,
        matched_rules=rules,
        stripped_chars=stripped,
        sanitized_text=sanitized,
    )


def scan_retrieved_chunk(text: str) -> InjectionSignal:
    """Scan a retrieved RAG chunk for indirect prompt injection.

    Thresholds: score ≥ 3 → block, score ∈ {1, 2} → flag, score == 0 → clean.
    `sanitized_text` always has invisibles and role markers stripped.
    Includes `instructional_density` check not used for user input.
    """
    sanitized, stripped = _normalize(text)
    sanitized = _strip_role_markers(sanitized)

    score, rules = _score_patterns(sanitized, include_density=True)

    if stripped > 0:
        score += 1
        rules.insert(0, "invisible_unicode")

    severity = _severity_chunk(score)

    return InjectionSignal(
        score=score,
        severity=severity,
        matched_rules=rules,
        stripped_chars=stripped,
        sanitized_text=sanitized,
    )
