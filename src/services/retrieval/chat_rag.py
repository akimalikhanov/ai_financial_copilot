"""RAG pipeline for chat: embed, parallel retrieve, fuse, hydrate, rerank, assemble."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING, Literal
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from src.observability import langfuse as lf_client
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

if TYPE_CHECKING:
    from langfuse import Langfuse


logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _span(
    lf: Langfuse | None,
    name: str,
    *,
    as_type: str = "span",
    input: object = None,
    **metadata: object,
):
    if lf is None:
        yield None
        return
    with lf.start_as_current_observation(
        as_type=as_type,  # type: ignore[arg-type]
        name=name,
        input=input,
        metadata=metadata or None,
    ) as obs:
        yield obs


def _to_hit(chunk: RetrievedChunk) -> RetrievalHit:
    return RetrievalHit(
        id=str(chunk.chunk_id),
        score=round(chunk.score, 4) if chunk.score is not None else None,
        vector_score=round(chunk.vector_score, 4) if chunk.vector_score is not None else None,
        keyword_score=round(chunk.keyword_score, 4) if chunk.keyword_score is not None else None,
    )


async def _retrieve_with_timeout(coro, timeout: float | None = None) -> list:
    """Run retrieval coroutine with timeout. Fail open: return [] on error/timeout."""
    timeout = timeout if timeout is not None else get_chat_retrieval_timeout()
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except (TimeoutError, Exception) as e:
        logger.warning("retrieval_backend_failed", extra={"error": str(e)})
        return []


async def _run_single_pass(
    semantic_vector: list[float],
    keyword_query: str,
    user_id: UUID,
    doc_ids: list[UUID] | None,
    timeout: float,
    vector_top_k: int,
    keyword_top_k: int,
    search_mode: Literal["hybrid", "vector", "keyword"] = "hybrid",
) -> tuple[list[RetrievedChunk], list[RetrievedChunk], list[RetrievedChunk]]:
    """Run retrieval backends in parallel (skipping one when search_mode is single-backend).

    Returns (vector_results, keyword_results, fused).
    For single-backend modes, fused == the single backend's results (no RRF).
    """
    if search_mode == "vector":
        vector_results = await _retrieve_with_timeout(
            qdrant_retrieve(semantic_vector, user_id, doc_ids=doc_ids, top_k=vector_top_k),
            timeout,
        )
        return vector_results, [], vector_results
    if search_mode == "keyword":
        keyword_results = await _retrieve_with_timeout(
            opensearch_retrieve(keyword_query, user_id, doc_ids=doc_ids, top_k=keyword_top_k),
            timeout,
        )
        return [], keyword_results, keyword_results

    vector_results, keyword_results = await asyncio.gather(
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
    return vector_results, keyword_results, fused


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

    lf = lf_client.get_client()

    with _span(lf, "embed_query", as_type="embedding", input=[transformed.semantic_query]) as obs:
        vectors_list = await asyncio.to_thread(embed_chunks, [transformed.semantic_query])
        if obs:
            obs.update(
                output={
                    "vector_count": len(vectors_list),
                    "dims": len(vectors_list[0]) if vectors_list else 0,
                }
            )
    semantic_vector = vectors_list[0]

    with _span(
        lf,
        "hybrid_retrieve",
        as_type="retriever",
        input={
            "semantic_query": transformed.semantic_query,
            "keyword_query": transformed.keyword_query,
            "search_mode": search_mode,
        },
        mode="single_pass",
    ) as obs:
        vec_r, kw_r, fused = await _run_single_pass(
            semantic_vector,
            transformed.keyword_query,
            user_id,
            doc_ids,
            timeout,
            vector_top_k,
            keyword_top_k,
            search_mode=search_mode,
        )
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
        )
        return RAGContext(formatted_context="", items=(), chunk_count=0), trace, []

    chunk_ids = [c.chunk_id for c in capped]
    payloads = await get_chunk_prompt_payloads(session, chunk_ids)
    texts_map = {cid: payloads[cid].prompt_text for cid in chunk_ids if cid in payloads}
    with _span(
        lf,
        "rerank",
        as_type="retriever",
        input={
            "query": transformed.semantic_query,
            "input_count": len(capped),
            "chunks": [{"chunk_id": str(c.chunk_id), "score": round(c.score, 4)} for c in capped],
        },
        mode="single_pass",
    ) as obs:
        reranked = await reranker.rerank(transformed.semantic_query, capped, texts_map)
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
    )
    with _span(
        lf,
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
