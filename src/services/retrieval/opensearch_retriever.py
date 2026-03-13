"""Async OpenSearch BM25 retrieval."""

from __future__ import annotations

import asyncio
import os
from typing import Any
from uuid import UUID

from src.schemas.retrieval import RetrievedChunk
from src.services.opensearch_client import get_client, reset_client
from src.utils.config import get_keyword_search_top_k

_INDEX_NAME = os.getenv("OPENSEARCH_INDEX", "chunks")


def _build_query(
    query_text: str, user_id: UUID, doc_ids: list[UUID] | None, top_k: int
) -> dict[str, Any]:
    filters: list[dict[str, Any]] = [{"term": {"user_id": str(user_id)}}]
    if doc_ids:
        filters.append({"terms": {"document_id": [str(doc_id) for doc_id in doc_ids]}})

    return {
        "size": top_k,
        "track_total_hits": False,
        "_source": [
            "chunk_id",
            "document_id",
            "chunk_index",
            "page_start",
            "page_end",
            "heading_trail",
            "chunk_type",
        ],
        "query": {
            "bool": {
                "must": [
                    {
                        "match": {
                            "enriched_text": {
                                "query": query_text,
                            }
                        }
                    }
                ],
                "filter": filters,
            }
        },
    }


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []

    max_score = max(scores)
    min_score = min(scores)
    if max_score <= 0:
        return [0.0] * len(scores)
    if max_score == min_score:
        return [1.0] * len(scores)
    scale = max_score - min_score
    return [(score - min_score) / scale for score in scores]


def _to_retrieved_chunk(
    hit: dict[str, Any],
    normalized_score: float,
    rank: int,
) -> RetrievedChunk | None:
    source = hit.get("_source") or {}
    chunk_id = source.get("chunk_id")
    document_id = source.get("document_id")
    if chunk_id is None or document_id is None:
        return None

    heading_trail = source.get("heading_trail")
    try:
        return RetrievedChunk(
            chunk_id=UUID(str(chunk_id)),
            document_id=UUID(str(document_id)),
            score=normalized_score,
            chunk_index=int(source.get("chunk_index", 0)),
            page_start=_coerce_int(source.get("page_start")),
            page_end=_coerce_int(source.get("page_end")),
            heading_trail=[str(item) for item in heading_trail]
            if isinstance(heading_trail, list)
            else [],
            source="keyword",
            chunk_type=str(source["chunk_type"]) if source.get("chunk_type") is not None else None,
            keyword_rank=rank,
            keyword_score=normalized_score,
        )
    except (TypeError, ValueError):
        return None


def _search(
    query_text: str,
    user_id: UUID,
    *,
    doc_ids: list[UUID] | None,
    top_k: int,
) -> list[RetrievedChunk]:
    response = get_client().search(
        index=_INDEX_NAME,
        body=_build_query(query_text, user_id, doc_ids, top_k),
    )
    hits = response.get("hits", {}).get("hits", [])
    scores = _normalize_scores([float(hit.get("_score") or 0.0) for hit in hits])

    chunks: list[RetrievedChunk] = []
    for rank, (hit, score) in enumerate(zip(hits, scores, strict=True), start=1):
        chunk = _to_retrieved_chunk(hit, score, rank)
        if chunk is not None:
            chunks.append(chunk)
    return chunks


async def retrieve(
    query_text: str,
    user_id: UUID,
    *,
    doc_ids: list[UUID] | None = None,
    top_k: int = get_keyword_search_top_k(),
) -> list[RetrievedChunk]:
    """Search OpenSearch for a user's most relevant chunks."""
    query_text = query_text.strip()
    if top_k <= 0 or not query_text:
        return []

    return await asyncio.to_thread(
        _search,
        query_text,
        user_id,
        doc_ids=doc_ids,
        top_k=top_k,
    )


__all__ = ["reset_client", "retrieve"]
