"""RAG pipeline for chat: embed, parallel retrieve, fuse, hydrate, rerank, assemble."""

from __future__ import annotations

import asyncio
import logging
from time import perf_counter
from typing import Literal
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from src.observability.langfuse import span as lf_span
from src.observability.metrics import RAG_CHUNKS, RAG_RETRIEVAL
from src.schemas.query_transform import TransformedQuery
from src.schemas.retrieval import RAGContext, RetrievalHit, RetrievalTrace, RetrievedChunk
from src.services.ingestion.embedder import embed_chunks
from src.services.retrieval.context_assembler import assemble_rag_context
from src.services.retrieval.hybrid_retriever import fuse_rrf
from src.services.retrieval.opensearch_retriever import retrieve as opensearch_retrieve
from src.services.retrieval.payload_hydrator import get_chunk_prompt_payloads
from src.services.retrieval.qdrant_retriever import retrieve as qdrant_retrieve
from src.services.retrieval.reranker import Reranker, get_reranker
from src.utils.config import (
    get_chat_retrieval_timeout,
    get_keyword_search_top_k,
    get_reranker_max_input,
    get_vector_search_top_k,
)

logger = logging.getLogger(__name__)


def _to_hit(chunk: RetrievedChunk) -> RetrievalHit:
    return RetrievalHit(
        id=str(chunk.chunk_id),
        score=round(chunk.score, 4) if chunk.score is not None else None,
        vector_score=round(chunk.vector_score, 4) if chunk.vector_score is not None else None,
        keyword_score=round(chunk.keyword_score, 4) if chunk.keyword_score is not None else None,
    )


async def _retrieve_with_timeout(coro, timeout: float | None = None) -> tuple[list, bool]:
    """Run retrieval coroutine with timeout.

    Fails open with ``[]`` so one dead backend cannot take down the request, but reports
    it via ``ok=False``: callers must be able to tell "the corpus does not discuss this"
    from "the index was unreachable" (P1-F), which an empty list alone cannot express.
    """
    timeout = timeout if timeout is not None else get_chat_retrieval_timeout()
    try:
        return await asyncio.wait_for(coro, timeout=timeout), True
    except TimeoutError as e:
        logger.warning("retrieval_backend_failed", extra={"error": str(e), "reason": "timeout"})
        return [], False
    except Exception as e:
        logger.warning("retrieval_backend_failed", extra={"error": str(e)})
        return [], False


async def _run_single_pass(
    semantic_vector: list[float],
    keyword_query: str,
    user_id: UUID,
    doc_ids: list[UUID] | None,
    timeout: float,
    vector_top_k: int,
    keyword_top_k: int,
    search_mode: Literal["hybrid", "vector", "keyword"] = "hybrid",
) -> tuple[list[RetrievedChunk], list[RetrievedChunk], list[RetrievedChunk], bool]:
    """Run retrieval backends in parallel (skipping one when search_mode is single-backend).

    Returns (vector_results, keyword_results, fused, all_backends_failed).
    For single-backend modes, fused == the single backend's results (no RRF), and
    all_backends_failed reflects that one backend alone.
    """
    if search_mode == "vector":
        vector_results, vec_ok = await _retrieve_with_timeout(
            qdrant_retrieve(semantic_vector, user_id, doc_ids=doc_ids, top_k=vector_top_k),
            timeout,
        )
        return vector_results, [], vector_results, not vec_ok
    if search_mode == "keyword":
        keyword_results, kw_ok = await _retrieve_with_timeout(
            opensearch_retrieve(keyword_query, user_id, doc_ids=doc_ids, top_k=keyword_top_k),
            timeout,
        )
        return [], keyword_results, keyword_results, not kw_ok

    (vector_results, vec_ok), (keyword_results, kw_ok) = await asyncio.gather(
        _retrieve_with_timeout(
            qdrant_retrieve(semantic_vector, user_id, doc_ids=doc_ids, top_k=vector_top_k),
            timeout,
        ),
        _retrieve_with_timeout(
            opensearch_retrieve(keyword_query, user_id, doc_ids=doc_ids, top_k=keyword_top_k),
            timeout,
        ),
    )
    fused = fuse_rrf(vector_results, keyword_results)
    # Only a total outage is reported: with one backend alive the request still has real
    # retrieval, and degrading it to "search unavailable" would be a false alarm.
    return vector_results, keyword_results, fused, not (vec_ok or kw_ok)


async def run_chat_rag_pipeline(
    session: AsyncSession,
    *,
    transformed: TransformedQuery,
    user_id: UUID,
    doc_ids: list[UUID] | None,
    timeout: float | None = None,
    reranker: Reranker | None = None,
    search_mode: Literal["hybrid", "vector", "keyword"] = "hybrid",
    top_k_override: int | None = None,
) -> tuple[RAGContext, RetrievalTrace, list[RetrievedChunk]]:
    """Embed query, run retrieval (single-pass), rerank, assemble RAGContext.

    search_mode controls which backends run:
      - "hybrid": Qdrant + OpenSearch in parallel, fused via RRF (default)
      - "vector":  Qdrant only, no OpenSearch, no RRF
      - "keyword": OpenSearch only, no Qdrant, no RRF
    """
    timeout = timeout if timeout is not None else get_chat_retrieval_timeout()
    vector_top_k = top_k_override if top_k_override is not None else get_vector_search_top_k()
    keyword_top_k = top_k_override if top_k_override is not None else get_keyword_search_top_k()
    reranker_max_input = get_reranker_max_input()

    if reranker is None:
        reranker = get_reranker()

    with lf_span("embed_query", as_type="embedding", input=[transformed.semantic_query]) as obs:
        _t = perf_counter()
        vectors_list = await asyncio.to_thread(embed_chunks, [transformed.semantic_query])
        RAG_RETRIEVAL.labels("embed").observe(perf_counter() - _t)
        if obs:
            obs.update(
                output={
                    "vector_count": len(vectors_list),
                    "dims": len(vectors_list[0]) if vectors_list else 0,
                }
            )
    semantic_vector = vectors_list[0]

    with lf_span(
        "hybrid_retrieve",
        as_type="retriever",
        input={
            "semantic_query": transformed.semantic_query,
            "keyword_query": transformed.keyword_query,
            "search_mode": search_mode,
        },
        mode="single_pass",
    ) as obs:
        _t = perf_counter()
        vec_r, kw_r, fused, all_backends_failed = await _run_single_pass(
            semantic_vector,
            transformed.keyword_query,
            user_id,
            doc_ids,
            timeout,
            vector_top_k,
            keyword_top_k,
            search_mode=search_mode,
        )
        RAG_RETRIEVAL.labels("hybrid_retrieve").observe(perf_counter() - _t)
        RAG_CHUNKS.labels("vector").observe(len(vec_r))
        RAG_CHUNKS.labels("keyword").observe(len(kw_r))
        RAG_CHUNKS.labels("fused").observe(len(fused))
        if obs:
            obs.update(
                output={
                    "counts": {"vector": len(vec_r), "keyword": len(kw_r), "fused": len(fused)},
                    "vector": [_to_hit(c).model_dump(exclude_none=True) for c in vec_r],
                    "keyword": [_to_hit(c).model_dump(exclude_none=True) for c in kw_r],
                    "fused": [_to_hit(c).model_dump(exclude_none=True) for c in fused],
                }
            )
    capped = fused[:reranker_max_input]
    if not capped:
        trace = RetrievalTrace(
            qdrant=[_to_hit(c) for c in vec_r],
            opensearch=[_to_hit(c) for c in kw_r],
            all_backends_failed=all_backends_failed,
        )
        return RAGContext(formatted_context="", items=(), chunk_count=0), trace, []

    chunk_ids = [c.chunk_id for c in capped]
    payloads = await get_chunk_prompt_payloads(session, chunk_ids)
    texts_map = {cid: payloads[cid].prompt_text for cid in chunk_ids if cid in payloads}
    with lf_span(
        "rerank",
        as_type="retriever",
        input={
            "query": transformed.semantic_query,
            "input_count": len(capped),
            "chunks": [{"chunk_id": str(c.chunk_id), "score": round(c.score, 4)} for c in capped],
        },
        mode="single_pass",
    ) as obs:
        _t = perf_counter()
        reranked = await reranker.rerank(transformed.semantic_query, capped, texts_map)
        RAG_RETRIEVAL.labels("rerank").observe(perf_counter() - _t)
        RAG_CHUNKS.labels("reranked").observe(len(reranked))
        if obs:
            obs.update(
                output={
                    "output_count": len(reranked),
                    "chunks": [
                        {"chunk_id": str(c.chunk_id), "score": round(c.score, 4)} for c in reranked
                    ],
                }
            )

    trace = RetrievalTrace(
        qdrant=[_to_hit(c) for c in vec_r],
        opensearch=[_to_hit(c) for c in kw_r],
        fused=[_to_hit(c) for c in fused],
        reranked=[_to_hit(c) for c in reranked],
        all_backends_failed=all_backends_failed,
    )
    with lf_span(
        "assemble_context",
        input=[{"chunk_id": str(c.chunk_id), "score": round(c.score, 4)} for c in reranked],
    ) as obs:
        ctx, guardrails = assemble_rag_context(reranked, payloads)
        if obs:
            obs.update(
                output={
                    "chunk_count": ctx.chunk_count,
                    "context_chars": len(ctx.formatted_context or ""),
                }
            )
    trace.dropped_chunks = guardrails.dropped
    trace.flagged_chunks = guardrails.flagged
    return ctx, trace, reranked
