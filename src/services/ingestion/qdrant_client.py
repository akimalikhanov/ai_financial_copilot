"""Qdrant client helpers for ingestion vectors."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any
from uuid import UUID


@lru_cache(maxsize=1)
def _get_client():
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise RuntimeError(
            "qdrant-client is required for vector storage. "
            "Install it with `.venv/bin/python -m pip install qdrant-client`."
        ) from exc

    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    return QdrantClient(host=host, port=port)


def ensure_collection(name: str, dim: int) -> None:
    """Create collection if it does not exist."""
    from qdrant_client.http.models import Distance, VectorParams

    client = _get_client()
    if client.collection_exists(collection_name=name):
        return

    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )


def upsert_chunks(
    collection: str,
    doc_id: UUID | str,
    chunks_with_vectors: list[dict[str, Any]],
) -> None:
    """
    Upsert chunk vectors with lean payload.

    Each item must have: vector, chunk_id (UUID), chunk_index.
    Optional: chunk_type, page_start, page_end, heading_trail.
    Text is fetched from Postgres at retrieval time.
    """
    if not chunks_with_vectors:
        return

    from qdrant_client.http.models import PointStruct

    doc_id_str = str(doc_id)
    points: list[PointStruct] = []

    for item in chunks_with_vectors:
        vector = item["vector"]
        chunk_id = item["chunk_id"]
        chunk_index = item["chunk_index"]

        payload: dict[str, Any] = {
            "chunk_id": str(chunk_id),
            "document_id": doc_id_str,
            "chunk_index": chunk_index,
        }
        for key in ("chunk_type", "page_start", "page_end", "heading_trail"):
            if key in item and item[key] is not None:
                payload[key] = item[key]

        points.append(PointStruct(id=str(chunk_id), vector=vector, payload=payload))

    _get_client().upsert(collection_name=collection, points=points, wait=True)


def delete_by_document(collection: str, doc_id: UUID | str) -> None:
    """Delete all vectors for a document."""
    from qdrant_client.http.models import FieldCondition, Filter, MatchValue

    _get_client().delete(
        collection_name=collection,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=str(doc_id)),
                )
            ]
        ),
        wait=True,
    )
