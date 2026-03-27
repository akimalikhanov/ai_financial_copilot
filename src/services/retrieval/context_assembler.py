"""Pure context assembly: dedup, budget, format chunks with citations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from uuid import UUID

from src.schemas.retrieval import (
    REF_PLACEHOLDER,
    ChunkPromptPayload,
    Citation,
    ContextItem,
    RAGContext,
    RetrievedChunk,
)


def _dedup_chunks(chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
    seen: set[UUID] = set()
    out: list[RetrievedChunk] = []
    for c in chunks:
        if c.chunk_id not in seen:
            seen.add(c.chunk_id)
            out.append(c)
    return out


def assemble_rag_context(
    chunks: Sequence[RetrievedChunk],
    payloads: Mapping[UUID, ChunkPromptPayload],
    *,
    assume_unique: bool = True,
) -> RAGContext:
    """Build RAGContext from reranked chunks and hydrated payloads."""
    if not assume_unique:
        selected = _dedup_chunks(chunks)
    else:
        selected = list(chunks)

    items: list[ContextItem] = []
    blocks: list[str] = []

    for idx, chunk in enumerate(selected):
        payload = payloads.get(chunk.chunk_id)
        if payload is None:
            raise ValueError(f"Missing ChunkPromptPayload for chunk_id={chunk.chunk_id}")

        ref_id = f"S{idx + 1}"
        prompt_text = payload.prompt_text.replace(REF_PLACEHOLDER, ref_id)

        citation = Citation(
            ref_id=ref_id,
            ref_index=idx + 1,
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            document_name=payload.document_name,
            filename=None,
            page_numbers=payload.page_numbers,
            heading_path=payload.heading_trail,
            snippet=payload.snippet,
        )

        items.append(
            ContextItem(
                ref_id=ref_id,
                chunk_id=chunk.chunk_id,
                score=chunk.score,
                prompt_text=prompt_text,
                citation=citation,
                provenance=payload.provenance,
            )
        )
        blocks.append(prompt_text)

    return RAGContext(
        formatted_context="\n\n".join(blocks),
        items=tuple(items),
        chunk_count=len(items),
    )
