"""OpenSearch client helpers for ingestion full-text index."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any
from uuid import UUID


@lru_cache(maxsize=1)
def _get_client():
    try:
        from opensearchpy import OpenSearch
    except ImportError as exc:
        raise RuntimeError(
            "opensearch-py is required for full-text indexing. "
            "Install it with `.venv/bin/python -m pip install opensearch-py`."
        ) from exc

    host = os.getenv("OPENSEARCH_BIND_HOST", "localhost")
    port = int(os.getenv("OPENSEARCH_HTTP_PORT", "9200"))
    return OpenSearch(
        hosts=[{"host": host, "port": port}],
        use_ssl=False,
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
    )


def ensure_index(name: str) -> None:
    """Create index with mappings if it does not exist."""
    client = _get_client()
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


def bulk_index(index: str, doc_id: UUID | str, chunks: list[dict[str, Any]]) -> None:
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
                "chunk_id": str(item["chunk_id"]),
                "chunk_index": item["chunk_index"],
                "enriched_text": item["enriched_text"],
                "heading_trail": item.get("heading_trail", []),
                "chunk_type": item.get("chunk_type"),
                "page_start": item.get("page_start"),
                "page_end": item.get("page_end"),
                "metadata": item.get("metadata", {}),
            },
        }
        for item in chunks
    ]

    bulk(_get_client(), actions, refresh=True)


def delete_by_document(index: str, doc_id: UUID | str) -> None:
    """Delete all indexed chunks for a document."""
    _get_client().delete_by_query(
        index=index,
        body={"query": {"term": {"document_id": str(doc_id)}}},
        params={"conflicts": "proceed", "refresh": "true"},
    )
