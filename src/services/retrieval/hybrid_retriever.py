"""Hybrid retrieval helpers based on reciprocal rank fusion."""

from __future__ import annotations

from dataclasses import replace
from uuid import UUID

from src.schemas.retrieval import RetrievedChunk
from src.utils.config import (
    get_fuse_rrf_final_top_k,
    get_fuse_rrf_k,
    get_fuse_rrf_keyword_weight,
    get_fuse_rrf_vector_weight,
)


def _accumulate_rrf(
    fused_scores: dict[UUID, float],
    chunks_by_id: dict[UUID, RetrievedChunk],
    results: list[RetrievedChunk],
    *,
    weight: float,
    k: int,
) -> None:
    seen_chunk_ids: set[UUID] = set()

    for rank, chunk in enumerate(results, start=1):
        if chunk.chunk_id in seen_chunk_ids:
            continue

        seen_chunk_ids.add(chunk.chunk_id)
        fused_scores[chunk.chunk_id] = fused_scores.get(chunk.chunk_id, 0.0) + (weight / (k + rank))
        chunks_by_id.setdefault(chunk.chunk_id, chunk)


def fuse_rrf(
    vector_results: list[RetrievedChunk],
    keyword_results: list[RetrievedChunk],
    *,
    vector_weight: float = get_fuse_rrf_vector_weight(),
    keyword_weight: float = get_fuse_rrf_keyword_weight(),
    k: int = get_fuse_rrf_k(),
    final_top_k: int = get_fuse_rrf_final_top_k(),
) -> list[RetrievedChunk]:
    """Fuse vector and keyword rankings with reciprocal rank fusion."""
    if final_top_k <= 0:
        return []
    if k < 0:
        raise ValueError("k must be non-negative")
    if vector_weight < 0 or keyword_weight < 0:
        raise ValueError("RRF weights must be non-negative")

    fused_scores: dict[UUID, float] = {}
    chunks_by_id: dict[UUID, RetrievedChunk] = {}

    vector_meta: dict[UUID, tuple[int | None, float | None]] = {}
    keyword_meta: dict[UUID, tuple[int | None, float | None]] = {}

    if vector_weight > 0 and vector_results:
        for rank, chunk in enumerate(vector_results, start=1):
            vector_meta[chunk.chunk_id] = (
                chunk.vector_rank or rank,
                chunk.vector_score or chunk.score,
            )
        _accumulate_rrf(
            fused_scores,
            chunks_by_id,
            vector_results,
            weight=vector_weight,
            k=k,
        )
    if keyword_weight > 0 and keyword_results:
        for rank, chunk in enumerate(keyword_results, start=1):
            keyword_meta[chunk.chunk_id] = (
                chunk.keyword_rank or rank,
                chunk.keyword_score or chunk.score,
            )
        _accumulate_rrf(
            fused_scores,
            chunks_by_id,
            keyword_results,
            weight=keyword_weight,
            k=k,
        )

    ranked_chunk_ids = sorted(fused_scores, key=fused_scores.__getitem__, reverse=True)

    return [
        _annotate_hybrid_chunk(
            chunks_by_id[chunk_id],
            fused_scores[chunk_id],
            vector_meta.get(chunk_id),
            keyword_meta.get(chunk_id),
        )
        for chunk_id in ranked_chunk_ids[:final_top_k]
    ]


def _annotate_hybrid_chunk(
    chunk: RetrievedChunk,
    fused_score: float,
    vector_info: tuple[int | None, float | None] | None,
    keyword_info: tuple[int | None, float | None] | None,
) -> RetrievedChunk:
    vector_rank, vector_score = vector_info or (None, None)
    keyword_rank, keyword_score = keyword_info or (None, None)

    return replace(
        chunk,
        score=fused_score,
        source="hybrid",
        vector_rank=vector_rank,
        vector_score=vector_score,
        keyword_rank=keyword_rank,
        keyword_score=keyword_score,
    )


__all__ = ["fuse_rrf"]
