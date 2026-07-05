"""Shared OpenSearch client helpers."""

from __future__ import annotations

import os
from functools import lru_cache


@lru_cache(maxsize=1)
def get_client():
    try:
        from opensearchpy import OpenSearch
    except ImportError as exc:
        raise RuntimeError(
            "opensearch-py is required for OpenSearch operations. "
            "Install it with `.venv/bin/python -m pip install opensearch-py`."
        ) from exc

    host = os.getenv("OPENSEARCH_HOST", "localhost")
    port = int(os.getenv("OPENSEARCH_HTTP_PORT", "9200"))
    return OpenSearch(
        hosts=[{"host": host, "port": port}],
        use_ssl=False,
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
        pool_maxsize=10,
    )


def reset_client() -> None:
    """Clear cached client (call after fork)."""
    get_client.cache_clear()
