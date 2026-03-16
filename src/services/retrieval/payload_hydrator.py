"""Hydrate chunk IDs into ChunkPromptPayload for context assembly."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from src.repository.chunk_repository import ChunkRepository
from src.repository.document_repository import DocumentRepository
from src.schemas.retrieval import (
    REF_PLACEHOLDER,
    BoundingBox,
    ChunkPromptPayload,
    ChunkProvenance,
    ProvenanceItem,
)


def _parse_provenance(raw: list[dict] | None) -> ChunkProvenance | None:
    """Convert raw chunk provenance (from DB) to ChunkProvenance for UI highlighting."""
    if not raw:
        return None
    items: list[ProvenanceItem] = []
    for p in raw:
        bbox = None
        if b := p.get("bbox"):
            # Docling uses l,t,r,b (left, top, right, bottom)
            bbox = BoundingBox(
                left=float(b.get("l", 0)),
                top=float(b.get("t", 0)),
                right=float(b.get("r", 0)),
                bottom=float(b.get("b", 0)),
                coord_origin=str(b.get("coord_origin", "BOTTOMLEFT")),
            )
        cs = p.get("charspan")
        charspan = tuple(cs) if isinstance(cs, (list, tuple)) and len(cs) == 2 else None
        items.append(
            ProvenanceItem(
                page_no=int(p.get("page_no", 0)),
                label=str(p.get("label", "text")),
                self_ref=p.get("self_ref"),
                charspan=charspan,
                bbox=bbox,
            )
        )
    return ChunkProvenance(
        filename=None,
        mimetype=None,
        binary_hash=None,
        page_span=None,
        doc_item_refs=(),
        items=tuple(items),
    )


def _page_numbers(page_start: int | None, page_end: int | None) -> tuple[int, ...]:
    if page_start is None:
        return ()
    if page_end is None or page_end == page_start:
        return (page_start,)
    return tuple(range(page_start, page_end + 1))


def _format_prompt_block(
    doc_name: str,
    page_numbers: tuple[int, ...],
    heading_trail: tuple[str, ...],
    enriched_text: str,
) -> str:
    page_str = f"p.{page_numbers[0]}" if page_numbers else ""
    section_str = " > ".join(heading_trail) if heading_trail else ""
    parts = [REF_PLACEHOLDER, doc_name, page_str, section_str]
    header = " | ".join(p for p in parts if p)
    return f"[{header}]\n{enriched_text}"


async def get_chunk_prompt_payloads(
    session: AsyncSession,
    chunk_ids: Sequence[UUID],
) -> dict[UUID, ChunkPromptPayload]:
    """Fetch chunks and documents, return chunk_id -> ChunkPromptPayload."""
    if not chunk_ids:
        return {}

    chunk_repo = ChunkRepository(session)
    doc_repo = DocumentRepository(session)

    chunks = await chunk_repo.get_by_ids(list(chunk_ids))
    if not chunks:
        return {}

    doc_ids = list({c.document_id for c in chunks})
    docs = await doc_repo.get_by_ids(doc_ids)
    doc_by_id = {d.id: d for d in docs}

    result: dict[UUID, ChunkPromptPayload] = {}
    for chunk in chunks:
        doc = doc_by_id.get(chunk.document_id)
        doc_name = (doc.extracted_title or doc.original_filename) if doc else "Unknown"

        page_numbers = _page_numbers(chunk.page_start, chunk.page_end)
        heading_trail = tuple(chunk.heading_trail or ())

        prompt_text = _format_prompt_block(
            doc_name=doc_name,
            page_numbers=page_numbers,
            heading_trail=heading_trail,
            enriched_text=chunk.enriched_text,
        )

        provenance = _parse_provenance(
            chunk.provenance if isinstance(chunk.provenance, list) else None
        )

        result[chunk.id] = ChunkPromptPayload(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            document_name=doc_name,
            page_numbers=page_numbers,
            heading_trail=heading_trail,
            prompt_text=prompt_text,
            snippet=None,
            provenance=provenance,
        )

    return result
