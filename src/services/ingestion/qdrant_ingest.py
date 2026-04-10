"""Qdrant client helpers for ingestion vectors."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import suppress
from typing import Any, cast
from uuid import UUID

from src.services.qdrant_client import get_client
from src.services.qdrant_client import reset_client as reset_shared_client


def ensure_collection(name: str, dim: int) -> None:
    """Create collection if it does not exist; create payload indexes for filtering."""
    from qdrant_client.http.models import Distance, PayloadSchemaType, VectorParams

    client = get_client()
    if not client.collection_exists(collection_name=name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
    for field in ("user_id", "document_id"):
        with suppress(Exception):
            client.create_payload_index(name, field, PayloadSchemaType.KEYWORD)


def upsert_chunks(
    collection: str,
    doc_id: UUID | str,
    chunks_with_vectors: list[dict[str, Any]],
    *,
    user_id: UUID | str,
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
            "user_id": str(user_id),
            "chunk_index": chunk_index,
        }
        for key in ("chunk_type", "page_start", "page_end", "heading_trail"):
            if key in item and item[key] is not None:
                payload[key] = item[key]

        points.append(PointStruct(id=str(chunk_id), vector=vector, payload=payload))

    # Qdrant's default max_request_size_mb is 32 MB. A 768-dim vector is ~12 KB
    # in JSON, so 500 points ≈ 6 MB per batch — well within the limit.
    client = get_client()
    for i in range(0, len(points), 500):
        client.upsert(collection_name=collection, points=points[i : i + 500], wait=True)


def delete_by_document(collection: str, doc_id: UUID | str) -> None:
    """Delete all vectors for a document."""
    from qdrant_client.http.models import FieldCondition, Filter, MatchValue

    get_client().delete(
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


def delete_by_chunk_ids(collection: str, chunk_ids: Sequence[UUID | str]) -> None:
    """Delete vectors by point IDs (chunk IDs)."""
    if not chunk_ids:
        return

    from qdrant_client.http.models import ExtendedPointId, PointIdsList

    ids = [str(cid) for cid in chunk_ids]
    get_client().delete(
        collection_name=collection,
        points_selector=PointIdsList(points=cast(list[ExtendedPointId], ids)),
        wait=True,
    )


def reset_client() -> None:
    """Clear cached client (call after fork)."""
    reset_shared_client()
