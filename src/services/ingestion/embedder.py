"""Embedding service for ingestion chunks."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import httpx

from src.utils.config import (
    get_embedder_base_url,
    get_embedder_batch_size,
    get_embedder_concurrency,
    get_embedder_timeout_seconds,
    get_embedding_device,
    get_embedding_dim,
    get_embedding_model,
    get_embedding_provider,
)

_LOG = logging.getLogger(__name__)

_st_lock = threading.Lock()


@lru_cache(maxsize=1)
def _get_sentence_transformer(model_name: str, device: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for local embeddings. "
            "Install it with `.venv/bin/python -m pip install sentence-transformers`."
        ) from exc
    return SentenceTransformer(model_name, device=device)


@lru_cache(maxsize=1)
def _get_openai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "openai package is required for OpenAI embeddings. "
            "Install it with `.venv/bin/python -m pip install openai`."
        ) from exc
    return OpenAI()


def reset_clients() -> None:
    """Clear cached clients/models (call after fork)."""
    global _tei_batch_size
    _get_sentence_transformer.cache_clear()
    _get_openai_client.cache_clear()
    with _tei_lock:
        _tei_batch_size = None


_tei_lock = threading.Lock()
_tei_batch_size: int | None = None


def _resolve_tei_batch_size(base_url: str, timeout: float) -> int:
    """EMBEDDER_BATCH_SIZE, clamped to what this TEI server actually accepts.

    The two have to agree: over --max-client-batch-size TEI rejects the request outright.
    Probing /info once per process makes an oversized config self-correct instead of failing
    at ingest time. Falls back to the configured value if /info is unreachable.
    """
    global _tei_batch_size
    if _tei_batch_size is not None:
        return _tei_batch_size
    with _tei_lock:
        if _tei_batch_size is not None:
            return _tei_batch_size
        configured = get_embedder_batch_size()
        resolved = configured
        try:
            info = httpx.get(f"{base_url}/info", timeout=timeout).json()
            server_max = int(info["max_client_batch_size"])
            resolved = min(configured, server_max)
            if resolved != configured:
                _LOG.warning(
                    "embedder.batch_size_clamped",
                    extra={"configured": configured, "server_max": server_max},
                )
        except Exception:
            _LOG.warning(
                "embedder.info_probe_failed",
                extra={"base_url": base_url, "batch_size": configured},
                exc_info=True,
            )
        _tei_batch_size = resolved
        return resolved


def _post_batch(client: httpx.Client, base_url: str, batch: list[str]) -> list[list[float]]:
    response = client.post(f"{base_url}/embed", json={"inputs": batch, "normalize": True})
    response.raise_for_status()
    return response.json()


def _embed_tei(chunks: list[str]) -> list[list[float]]:
    base_url = get_embedder_base_url().rstrip("/")
    timeout = get_embedder_timeout_seconds()
    batch_size = _resolve_tei_batch_size(base_url, timeout)
    batches = [chunks[i : i + batch_size] for i in range(0, len(chunks), batch_size)]
    # TEI queues each input separately and batches across requests, so one in-flight request
    # leaves the GPU idle between round-trips. Concurrency keeps its queue non-empty; the
    # ceiling is --max-concurrent-requests (512), far above anything we send.
    concurrency = min(get_embedder_concurrency(), len(batches))
    with httpx.Client(
        timeout=timeout, limits=httpx.Limits(max_connections=max(concurrency, 1))
    ) as client:
        if concurrency <= 1:
            batch_vectors = [_post_batch(client, base_url, b) for b in batches]
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                # .map preserves input order, so vectors stay aligned with chunks, and
                # re-raises the first batch failure when the results are consumed.
                batch_vectors = list(pool.map(lambda b: _post_batch(client, base_url, b), batches))
    return [vector for batch in batch_vectors for vector in batch]


def _embed_local(chunks: list[str], model_name: str) -> list[list[float]]:
    with _st_lock:
        model = _get_sentence_transformer(model_name, get_embedding_device())
    vectors = model.encode(chunks, batch_size=32, convert_to_numpy=True, show_progress_bar=False)
    return vectors.tolist()


def _embed_openai(chunks: list[str], model_name: str) -> list[list[float]]:
    client = _get_openai_client()
    response = client.embeddings.create(model=model_name, input=chunks)
    return [list(item.embedding) for item in response.data]


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """Batch-embed chunk texts using TEI, OpenAI, or local SentenceTransformer."""
    if not chunks:
        return []

    provider = get_embedding_provider()
    model_name = get_embedding_model()

    if provider == "tei":
        vectors = _embed_tei(chunks)
    elif provider == "openai":
        vectors = _embed_openai(chunks, model_name)
    else:
        vectors = _embed_local(chunks, model_name)

    expected_dim = get_embedding_dim()
    if expected_dim is not None and any(len(v) != expected_dim for v in vectors):
        actual = len(vectors[0]) if vectors else 0
        raise RuntimeError(
            f"Embedding dimension mismatch: expected {expected_dim}, got {actual} from provider '{provider}'"
        )

    return vectors
