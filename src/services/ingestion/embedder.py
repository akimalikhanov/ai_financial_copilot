"""Embedding service for ingestion chunks."""

from __future__ import annotations

from functools import lru_cache

from src.utils.config import get_embedding_dim, get_embedding_model, get_embedding_provider


@lru_cache(maxsize=1)
def _get_sentence_transformer(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for local embeddings. "
            "Install it with `.venv/bin/python -m pip install sentence-transformers`."
        ) from exc
    return SentenceTransformer(model_name)


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
    _get_sentence_transformer.cache_clear()
    _get_openai_client.cache_clear()


def _embed_local(chunks: list[str], model_name: str) -> list[list[float]]:
    model = _get_sentence_transformer(model_name)
    vectors = model.encode(chunks, batch_size=32, convert_to_numpy=True, show_progress_bar=False)
    return vectors.tolist()


def _embed_openai(chunks: list[str], model_name: str) -> list[list[float]]:
    client = _get_openai_client()
    response = client.embeddings.create(model=model_name, input=chunks)
    return [list(item.embedding) for item in response.data]


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """Batch-embed chunk texts using local model or OpenAI API."""
    if not chunks:
        return []

    provider = get_embedding_provider()
    model_name = get_embedding_model()

    if provider == "openai":
        vectors = _embed_openai(chunks, model_name)
    else:
        vectors = _embed_local(chunks, model_name)

    expected_dim = get_embedding_dim()
    if expected_dim is not None and any(len(v) != expected_dim for v in vectors):
        actual = len(vectors[0]) if vectors else 0
        raise RuntimeError(
            f"Embedding dimension mismatch: expected {expected_dim}, got {actual} from model '{model_name}'"
        )

    return vectors
