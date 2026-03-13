"""OpenSearch client helpers for ingestion full-text index."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from uuid import UUID

from src.services.opensearch_client import get_client
from src.services.opensearch_client import reset_client as reset_shared_client


def ensure_index(name: str) -> None:
    """Create index with mappings if it does not exist."""
    client = get_client()
    if client.indices.exists(index=name):
        return

    client.indices.create(
        index=name,
        body={
            "settings": {
                "index": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                }
            },
            "mappings": {
                "properties": {
                    "document_id": {"type": "keyword"},
                    "user_id": {"type": "keyword"},
                    "chunk_id": {"type": "keyword"},
                    "chunk_index": {"type": "integer"},
                    "enriched_text": {"type": "text"},
                    "heading_trail": {"type": "keyword"},
                    "chunk_type": {"type": "keyword"},
                    "page_start": {"type": "integer"},
                    "page_end": {"type": "integer"},
                    "metadata": {"type": "object"},
                }
            },
        },
    )


def bulk_index(
    index: str,
    doc_id: UUID | str,
    chunks: list[dict[str, Any]],
    *,
    user_id: UUID | str,
) -> None:
    """Bulk-index document chunks for full-text retrieval."""
    if not chunks:
        return

    try:
        from opensearchpy.helpers import bulk
    except ImportError as exc:
        raise RuntimeError(
            "opensearch-py is required for full-text indexing. "
            "Install it with `.venv/bin/python -m pip install opensearch-py`."
        ) from exc

    doc_id_str = str(doc_id)
    actions = [
        {
            "_op_type": "index",
            "_index": index,
            "_id": str(item["chunk_id"]),
            "_source": {
                "document_id": doc_id_str,
                "user_id": str(user_id),
                "chunk_id": str(item["chunk_id"]),
                "chunk_index": item["chunk_index"],
                "enriched_text": item["enriched_text"],
                "heading_trail": item.get("heading_trail", []),
                "chunk_type": item.get("chunk_type"),
                "page_start": item.get("page_start"),
                "page_end": item.get("page_end"),
            },
        }
        for item in chunks
    ]

    bulk(get_client(), actions, refresh=True)


def delete_by_document(index: str, doc_id: UUID | str) -> None:
    """Delete all indexed chunks for a document."""
    get_client().delete_by_query(
        index=index,
        body={"query": {"term": {"document_id": str(doc_id)}}},
        params={"conflicts": "proceed"},
    )


def bulk_delete(index: str, chunk_ids: Sequence[UUID | str]) -> None:
    """Bulk-delete chunks by _id (chunk_id)."""
    if not chunk_ids:
        return

    try:
        from opensearchpy.helpers import bulk
    except ImportError as exc:
        raise RuntimeError(
            "opensearch-py is required for full-text indexing. "
            "Install it with `.venv/bin/python -m pip install opensearch-py`."
        ) from exc

    actions = [{"_op_type": "delete", "_index": index, "_id": str(cid)} for cid in chunk_ids]
    bulk(get_client(), actions, refresh=True)


def reset_client() -> None:
    """Clear cached client (call after fork)."""
    reset_shared_client()
