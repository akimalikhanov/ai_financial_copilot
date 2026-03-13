"""Shared Qdrant client helpers."""

from __future__ import annotations

import os
from functools import lru_cache


@lru_cache(maxsize=1)
def get_client():
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise RuntimeError(
            "qdrant-client is required for Qdrant operations. "
            "Install it with `.venv/bin/python -m pip install qdrant-client`."
        ) from exc

    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    return QdrantClient(host=host, port=port)


def reset_client() -> None:
    """Clear cached client (call after fork)."""
    get_client.cache_clear()
