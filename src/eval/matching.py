from __future__ import annotations

import logging
import re
from uuid import UUID

from rapidfuzz import fuzz, process
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.document import Document

logger = logging.getLogger(__name__)

_STRIP_CHARS = re.compile(r"[.,;:()]")
_WHITESPACE = re.compile(r"\s+")
_FILE_EXT = re.compile(r"\.(pdf|docx?|xlsx?|csv|txt)$", re.IGNORECASE)

FUZZY_THRESHOLD = 90


def normalize_title(s: str) -> str:
    s = _FILE_EXT.sub("", s.strip())
    s = s.casefold()
    s = _STRIP_CHARS.sub("", s)
    return _WHITESPACE.sub(" ", s).strip()


def page_key(doc_name_or_id: str, page: int) -> str:
    return f"{normalize_title(doc_name_or_id)}:p{page}"


async def resolve_doc_ids(
    pools: list[list[str]],
    session: AsyncSession,
    user_id: UUID,
) -> dict[str, UUID | None]:
    """Map each unique golden title (from reference_pools entries) to a Document UUID.

    Each pool entry is "<Document Title>:p<N>". Titles are fuzzy-matched against
    Document.extracted_title (or original_filename as fallback) with threshold >= 90.
    Returns {raw_title: UUID | None}; None means unresolved.
    """
    raw_entries: set[str] = {entry for pool in pools for entry in pool}
    golden_titles: dict[str, str] = {}  # raw_entry -> extracted title part
    for entry in raw_entries:
        title_part = re.sub(r":p\d+$", "", entry)
        golden_titles[entry] = title_part

    result = await session.execute(
        select(Document.id, Document.extracted_title, Document.original_filename).where(
            Document.user_id == user_id
        )
    )
    rows = result.all()

    # Build candidate list: one entry per (doc_id, normalized_name) pair.
    # Each document contributes up to two candidates: extracted_title and original_filename.
    # This ensures golden titles written to match the filename resolve even when
    # extracted_title is a generic string like "Annual Report 2022".
    candidates: list[tuple[UUID, str]] = []  # (doc_id, normalized_name)
    for doc_id, extracted, filename in rows:
        seen: set[str] = set()
        for name in (extracted, filename):
            if name:
                norm = normalize_title(name)
                if norm not in seen:
                    seen.add(norm)
                    candidates.append((doc_id, norm))

    if not candidates:
        logger.warning("No documents found for user_id=%s", user_id)
        return dict.fromkeys(raw_entries)

    candidate_keys = [c[1] for c in candidates]
    resolver: dict[str, UUID | None] = {}

    for raw_entry, title in golden_titles.items():
        normalized = normalize_title(title)
        match = process.extractOne(
            normalized,
            candidate_keys,
            scorer=fuzz.ratio,
            score_cutoff=FUZZY_THRESHOLD,
        )
        if match is None:
            logger.warning("Unresolved golden title: %r (normalized: %r)", title, normalized)
            resolver[raw_entry] = None
        else:
            _, _, idx = match
            resolver[raw_entry] = candidates[idx][0]

    return resolver


def expand_pools_to_page_keys(
    pools: list[list[str]],
    resolver: dict[str, UUID | None],
    page_tolerance: int = 2,
) -> list[set[str]]:
    """Convert reference_pools (title:pN strings) into page_key sets per pool.

    Outer list = AND (all pools must be hit).
    Inner set = OR (any page_key in pool suffices).
    Entries whose title resolved to None are dropped from the pool.
    page_tolerance: accept ±N pages around each golden page number.
    """
    expanded: list[set[str]] = []
    for pool in pools:
        keys: set[str] = set()
        for entry in pool:
            if resolver.get(entry) is None:
                continue
            title_part = re.sub(r":p\d+$", "", entry)
            page_match = re.search(r":p(\d+)$", entry)
            if page_match:
                p = int(page_match.group(1))
                for offset in range(-page_tolerance, page_tolerance + 1):
                    if p + offset >= 1:
                        keys.add(page_key(title_part, p + offset))
        if keys:
            expanded.append(keys)
    return expanded
