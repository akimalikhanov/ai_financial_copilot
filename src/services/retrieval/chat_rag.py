"""RAG pipeline for chat: embed, parallel retrieve, fuse, hydrate, rerank, assemble."""

from __future__ import annotations

import asyncio
import logging
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from src.schemas.retrieval import RAGContext
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
# Hardcoded for tests
_DEFAULT_DOC_ID = UUID("9e1284e3-ef0b-49e6-a1b9-a9292a7e0fa2")


# DEPRECATED: replaced by src.services.router.scope_resolver.resolve_scope + ChatPipelineState.scope_result
def resolve_doc_ids(_user_id: UUID) -> list[UUID]:
    """Resolve document IDs for retrieval. For now returns a single hard-coded UUID."""
    return [_DEFAULT_DOC_ID]


async def _retrieve_with_timeout(coro, timeout: float | None = None) -> list:
    """Run retrieval coroutine with timeout. Fail open: return [] on error/timeout."""
    timeout = timeout if timeout is not None else get_chat_retrieval_timeout()
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except (TimeoutError, Exception) as e:
        logger.warning("retrieval_backend_failed", extra={"error": str(e)})
        return []


async def run_chat_rag_pipeline(
    session: AsyncSession,
    query: str,
    user_id: UUID,
    doc_ids: list[UUID] | None,
    timeout: float | None = None,
    reranker: Reranker | None = None,
) -> RAGContext:
    """Embed query, run parallel Qdrant + OpenSearch, fuse RRF, hydrate, rerank, assemble."""
    timeout = timeout if timeout is not None else get_chat_retrieval_timeout()
    query_vector = (await asyncio.to_thread(embed_chunks, [query]))[0]
    vector_top_k = get_vector_search_top_k()
    keyword_top_k = get_keyword_search_top_k()

    async def qdrant_task():
        return await qdrant_retrieve(query_vector, user_id, doc_ids=doc_ids, top_k=vector_top_k)

    async def opensearch_task():
        return await opensearch_retrieve(query, user_id, doc_ids=doc_ids, top_k=keyword_top_k)

    vector_results, keyword_results = await asyncio.gather(
        _retrieve_with_timeout(qdrant_task(), timeout),
        _retrieve_with_timeout(opensearch_task(), timeout),
    )

    fused = fuse_rrf(vector_results, keyword_results)
    capped = fused[: get_reranker_max_input()]
    if not capped:
        return RAGContext(formatted_context="", items=(), chunk_count=0)

    chunk_ids = [c.chunk_id for c in capped]
    payloads = await get_chunk_prompt_payloads(session, chunk_ids)
    texts = {cid: payloads[cid].prompt_text for cid in chunk_ids if cid in payloads}

    if reranker is None:
        reranker = get_reranker()
    reranked = await reranker.rerank(query, capped, texts)

    return assemble_rag_context(reranked, payloads)
