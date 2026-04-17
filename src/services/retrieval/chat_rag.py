"""RAG pipeline for chat: embed, parallel retrieve, fuse, hydrate, rerank, assemble."""

from __future__ import annotations

import asyncio
import logging
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

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
    get_multi_pass_chunks_per_sub,
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


async def _retrieve_with_timeout(coro, timeout: float | None = None) -> list:
    """Run retrieval coroutine with timeout. Fail open: return [] on error/timeout."""
    timeout = timeout if timeout is not None else get_chat_retrieval_timeout()
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except (TimeoutError, Exception) as e:
        logger.warning("retrieval_backend_failed", extra={"error": str(e)})
        return []


def _round_robin_interleave(
    ranked_lists: list[list[RetrievedChunk]],
    cap: int,
) -> list[RetrievedChunk]:
    """Take one chunk per sub-pass in turn, skipping already-seen chunk_ids."""
    seen: set[UUID] = set()
    result: list[RetrievedChunk] = []
    iters = [iter(lst) for lst in ranked_lists]
    while len(result) < cap:
        advanced = False
        for it in iters:
            if len(result) >= cap:
                break
            try:
                chunk = next(it)
                if chunk.chunk_id not in seen:
                    result.append(chunk)
                    seen.add(chunk.chunk_id)
                    advanced = True
            except StopIteration:
                pass
        if not advanced:
            break
    return result


async def _run_single_pass(
    semantic_vector: list[float],
    keyword_query: str,
    user_id: UUID,
    doc_ids: list[UUID] | None,
    timeout: float,
    vector_top_k: int,
    keyword_top_k: int,
) -> tuple[list[RetrievedChunk], list[RetrievedChunk], list[RetrievedChunk]]:
    """Run Qdrant + OpenSearch in parallel, fuse. Returns (vector, keyword, fused)."""
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
    sub_doc_ids: list[list[UUID] | None] | None = None,
    timeout: float | None = None,
    reranker: Reranker | None = None,
) -> tuple[RAGContext, RetrievalTrace]:
    """Embed queries, run hybrid retrieval, rerank, assemble RAGContext.

    Multi-pass mode (when transformed.sub_queries is non-empty):
      - sub_doc_ids must have the same length as transformed.sub_queries
      - Each sub-pass is scoped to its own doc_ids, reranked with sub.semantic_query
      - Results are round-robin interleaved and capped to reranker_top_k
      - All-empty fallback: runs single pass with top-level queries over doc_ids
    Single-pass mode: fuse_rrf → rerank(transformed.semantic_query) → assemble
    """
    timeout = timeout if timeout is not None else get_chat_retrieval_timeout()
    vector_top_k = get_vector_search_top_k()
    keyword_top_k = get_keyword_search_top_k()
    reranker_max_input = get_reranker_max_input()
    chunks_per_sub = get_multi_pass_chunks_per_sub()

    if reranker is None:
        reranker = get_reranker()

    sub_queries = transformed.sub_queries

    # --- Batch embed: top-level + all sub semantic queries, dedup ---
    all_semantic_texts = [transformed.semantic_query] + [s.semantic_query for s in sub_queries]
    unique_texts = list(dict.fromkeys(all_semantic_texts))  # preserve order, dedup
    vectors_list = await asyncio.to_thread(embed_chunks, unique_texts)
    text_to_vector: dict[str, list[float]] = dict(zip(unique_texts, vectors_list, strict=True))
    top_level_vector = text_to_vector[transformed.semantic_query]

    # ---------------------------------------------------------------
    # MULTI-PASS MODE
    # ---------------------------------------------------------------
    if sub_queries and sub_doc_ids is not None and len(sub_doc_ids) == len(sub_queries):

        async def _sub_pass(
            idx: int,
        ) -> tuple[list[RetrievedChunk], list[RetrievedChunk], list[RetrievedChunk]]:
            sub = sub_queries[idx]
            vec = text_to_vector[sub.semantic_query]
            sdoc = sub_doc_ids[idx]
            return await _run_single_pass(
                vec, sub.keyword_query, user_id, sdoc, timeout, vector_top_k, keyword_top_k
            )

        pass_results = await asyncio.gather(*[_sub_pass(i) for i in range(len(sub_queries))])

        # Collect all chunk_ids across passes for bulk hydration
        all_chunk_ids: list[UUID] = []
        for _, _, fused in pass_results:
            capped = fused[:reranker_max_input]
            all_chunk_ids.extend(c.chunk_id for c in capped)
        # Dedup preserving order for single DB call
        seen_ids: set[UUID] = set()
        unique_chunk_ids = [
            cid for cid in all_chunk_ids if not (cid in seen_ids or seen_ids.add(cid))
        ]  # type: ignore[func-returns-value]
        payloads = await get_chunk_prompt_payloads(session, unique_chunk_ids)
        texts_map: dict[UUID, str] = {
            cid: payloads[cid].prompt_text for cid in unique_chunk_ids if cid in payloads
        }

        # Per-sub rerank
        sub_pass_trace: list[dict] = []
        reranked_per_sub: list[list[RetrievedChunk]] = []

        async def _rerank_sub(idx: int) -> list[RetrievedChunk]:
            sub = sub_queries[idx]
            _, _, fused = pass_results[idx]
            capped = fused[:reranker_max_input]
            if not capped:
                return []
            reranked = await reranker.rerank(sub.semantic_query, capped, texts_map)
            return reranked[:chunks_per_sub]

        reranked_per_sub = list(
            await asyncio.gather(*[_rerank_sub(i) for i in range(len(sub_queries))])
        )

        for i, (sub, (vec_r, kw_r, fused_r)) in enumerate(
            zip(sub_queries, pass_results, strict=True)
        ):
            sub_pass_trace.append(
                {
                    "focus_entity": sub.focus_entity,
                    "entity_match_quality": sub.entity_match_quality,
                    "qdrant_hits": len(vec_r),
                    "opensearch_hits": len(kw_r),
                    "fused_hits": len(fused_r),
                    "reranked_hits": len(reranked_per_sub[i]),
                }
            )

        # Round-robin interleave, dedup, cap
        context_max = chunks_per_sub * len(sub_queries)
        interleaved = _round_robin_interleave(reranked_per_sub, cap=context_max)

        # All-empty fallback: every sub-pass returned nothing
        if not interleaved:
            logger.warning(
                "multi_pass_all_empty_fallback",
                extra={"user_id": str(user_id), "sub_count": len(sub_queries)},
            )
            vec_r, kw_r, fused = await _run_single_pass(
                top_level_vector,
                transformed.keyword_query,
                user_id,
                doc_ids,
                timeout,
                vector_top_k,
                keyword_top_k,
            )
            capped = fused[:reranker_max_input]
            if not capped:
                trace = RetrievalTrace(sub_passes=sub_pass_trace)
                return RAGContext(formatted_context="", items=(), chunk_count=0), trace
            fallback_ids = [c.chunk_id for c in capped]
            fallback_payloads = await get_chunk_prompt_payloads(session, fallback_ids)
            fallback_texts = {
                cid: fallback_payloads[cid].prompt_text
                for cid in fallback_ids
                if cid in fallback_payloads
            }
            interleaved = await reranker.rerank(transformed.semantic_query, capped, fallback_texts)
            payloads = fallback_payloads

        trace = RetrievalTrace(sub_passes=sub_pass_trace)
        return assemble_rag_context(interleaved, payloads, assume_unique=False), trace

    # ---------------------------------------------------------------
    # SINGLE-PASS MODE
    # ---------------------------------------------------------------
    vec_r, kw_r, fused = await _run_single_pass(
        top_level_vector,
        transformed.keyword_query,
        user_id,
        doc_ids,
        timeout,
        vector_top_k,
        keyword_top_k,
    )
    capped = fused[:reranker_max_input]
    if not capped:
        trace = RetrievalTrace(
            qdrant=[_to_hit(c) for c in vec_r],
            opensearch=[_to_hit(c) for c in kw_r],
        )
        return RAGContext(formatted_context="", items=(), chunk_count=0), trace

    chunk_ids = [c.chunk_id for c in capped]
    payloads = await get_chunk_prompt_payloads(session, chunk_ids)
    texts_map = {cid: payloads[cid].prompt_text for cid in chunk_ids if cid in payloads}
    reranked = await reranker.rerank(transformed.semantic_query, capped, texts_map)

    trace = RetrievalTrace(
        qdrant=[_to_hit(c) for c in vec_r],
        opensearch=[_to_hit(c) for c in kw_r],
        fused=[_to_hit(c) for c in fused],
        reranked=[_to_hit(c) for c in reranked],
    )
    return assemble_rag_context(reranked, payloads), trace
