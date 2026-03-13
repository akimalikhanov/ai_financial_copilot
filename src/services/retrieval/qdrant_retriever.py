"""Async Qdrant vector retrieval."""

from __future__ import annotations

import asyncio
import os
from typing import Any
from uuid import UUID

from src.schemas.retrieval import RetrievedChunk
from src.services.qdrant_client import get_client, reset_client
from src.utils.config import get_vector_search_top_k

_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "documents")


def _build_filter(user_id: UUID, doc_ids: list[UUID] | None):
    from qdrant_client.http.models import Condition, FieldCondition, Filter, MatchAny, MatchValue

    must: list[Condition] = [FieldCondition(key="user_id", match=MatchValue(value=str(user_id)))]
    if doc_ids:
        must.append(
            FieldCondition(
                key="document_id",
                match=MatchAny(any=[str(doc_id) for doc_id in doc_ids]),
            )
        )
    return Filter(must=must)


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _to_retrieved_chunk(point: Any, rank: int) -> RetrievedChunk | None:
    payload = point.payload or {}
    chunk_id = payload.get("chunk_id") or point.id
    document_id = payload.get("document_id")
    if chunk_id is None or document_id is None:
        return None

    heading_trail = payload.get("heading_trail")
    try:
        return RetrievedChunk(
            chunk_id=UUID(str(chunk_id)),
            document_id=UUID(str(document_id)),
            score=float(point.score),
            chunk_index=int(payload.get("chunk_index", 0)),
            page_start=_coerce_int(payload.get("page_start")),
            page_end=_coerce_int(payload.get("page_end")),
            heading_trail=[str(item) for item in heading_trail]
            if isinstance(heading_trail, list)
            else [],
            source="vector",
            chunk_type=str(payload["chunk_type"])
            if payload.get("chunk_type") is not None
            else None,
            vector_rank=rank,
            vector_score=float(point.score),
        )
    except (TypeError, ValueError):
        return None


def _search(
    query_vector: list[float],
    user_id: UUID,
    *,
    doc_ids: list[UUID] | None,
    top_k: int,
) -> list[RetrievedChunk]:
    response = get_client().query_points(
        collection_name=_COLLECTION_NAME,
        query=query_vector,
        query_filter=_build_filter(user_id, doc_ids),
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )

    chunks: list[RetrievedChunk] = []
    for rank, point in enumerate(response.points, start=1):
        chunk = _to_retrieved_chunk(point, rank)
        if chunk is not None:
            chunks.append(chunk)
    return chunks


async def retrieve(
    query_vector: list[float],
    user_id: UUID,
    *,
    doc_ids: list[UUID] | None = None,
    top_k: int = get_vector_search_top_k(),
) -> list[RetrievedChunk]:
    """Search Qdrant for a user's most relevant chunks."""
    if top_k <= 0 or not query_vector:
        return []

    return await asyncio.to_thread(
        _search,
        query_vector,
        user_id,
        doc_ids=doc_ids,
        top_k=top_k,
    )


__all__ = ["reset_client", "retrieve"]
