"""Cross-encoder reranking after fusion, before context assembly."""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, replace
from typing import Protocol
from uuid import UUID

import httpx

from src.schemas.retrieval import RetrievedChunk
from src.utils.config import (
    get_reranker_base_url,
    get_reranker_enabled,
    get_reranker_max_input,
    get_reranker_model_name,
    get_reranker_timeout_seconds,
    get_reranker_top_k,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RerankOutcome:
    """Reranked chunks plus what actually happened to produce them.

    ``scored`` is the load-bearing field: when it is False the chunks carry **fusion**
    scores (RRF, ~0.05 at k=20), not cross-encoder scores (~0–1). Comparing the two
    against one set of thresholds silently reports a ranking outage as weak grounding,
    so callers must branch on this rather than on the score alone.

    ``degraded`` separates "reranking is switched off" (NoOp — configured, expected)
    from "reranking broke" (fell open — a system fault worth surfacing).
    """

    chunks: list[RetrievedChunk]
    scored: bool
    degraded: bool = False


class Reranker(Protocol):
    """Protocol for reranking retrieved chunks."""

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        texts: dict[UUID, str],
    ) -> RerankOutcome: ...

    async def aclose(self) -> None:
        pass


class NoOpReranker:
    """Pass-through reranker for disabled reranking or fallback."""

    async def rerank(
        self,
        query: str,  # noqa: ARG002
        chunks: list[RetrievedChunk],
        texts: dict[UUID, str],  # noqa: ARG002
    ) -> RerankOutcome:
        # Not degraded: reranking being off is a configuration, not a failure.
        return RerankOutcome(chunks=list(chunks), scored=False)

    async def aclose(self) -> None:
        pass


class LocalCrossEncoderReranker:
    """TEI-backed cross-encoder reranker."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model_name: str | None = None,
        timeout_seconds: float | None = None,
        rerank_top_k: int | None = None,
        max_input: int | None = None,
    ) -> None:
        self._base_url = (base_url or get_reranker_base_url()).rstrip("/")
        self._model_name = model_name or get_reranker_model_name()
        self._timeout = (
            timeout_seconds if timeout_seconds is not None else get_reranker_timeout_seconds()
        )
        self._top_k = rerank_top_k if rerank_top_k is not None else get_reranker_top_k()
        self._max_input = max_input if max_input is not None else get_reranker_max_input()
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        texts: dict[UUID, str],
    ) -> RerankOutcome:
        if not chunks:
            return RerankOutcome(chunks=[], scored=True)
        to_rerank = chunks[: self._max_input]
        text_list = [texts.get(c.chunk_id, "") for c in to_rerank]
        if not any(text_list):
            # No payload text to score against — a hydration gap, not a reranker fault.
            return RerankOutcome(chunks=list(to_rerank), scored=False)

        def _fell_open() -> RerankOutcome:
            return RerankOutcome(chunks=list(chunks[: self._top_k]), scored=False, degraded=True)

        try:
            resp = await self._client.post(
                "/rerank",
                json={"query": query, "texts": text_list},
            )
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            logger.warning("reranker_failed", extra={"error": str(exc) or type(exc).__name__})
            return _fell_open()
        except Exception as exc:
            logger.warning("reranker_failed", extra={"error": str(exc) or type(exc).__name__})
            return _fell_open()

        if not isinstance(data, list):
            logger.warning("reranker_invalid_response", extra={"type": type(data).__name__})
            return _fell_open()

        score_by_idx: dict[int, float] = {}
        for item in data:
            if isinstance(item, dict) and "index" in item and "score" in item:
                idx = item["index"]
                with contextlib.suppress(TypeError, ValueError):
                    score_by_idx[int(idx)] = float(item["score"])

        if len(score_by_idx) != len(to_rerank):
            logger.warning(
                "reranker_incomplete_scores",
                extra={"expected": len(to_rerank), "got": len(score_by_idx)},
            )
            return _fell_open()

        ranked = [(to_rerank[i], score_by_idx.get(i, 0.0)) for i in range(len(to_rerank))]
        ranked.sort(key=lambda x: x[1], reverse=True)
        reranked = [replace(c, score=s) for c, s in ranked[: self._top_k]]
        return RerankOutcome(chunks=reranked, scored=True)


def get_reranker() -> Reranker:
    """Return enabled reranker or NoOpReranker when disabled."""
    if get_reranker_enabled():
        return LocalCrossEncoderReranker()
    return NoOpReranker()
