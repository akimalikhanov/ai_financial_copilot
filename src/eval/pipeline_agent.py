"""Eval pipeline variant that exercises the agentic retrieval path.

The original pipeline.py (single-pass RAG) is preserved for backward-compatibility.
This module mirrors its public interface — run_one returns a PipelineResult — but
internally drives run_agent_loop the same way the Celery chat task does.

Key differences from pipeline.py:
- Uses run_agent_loop (multi-turn tool-calling) instead of a single rewrite+retrieve.
- Requires a Redis connection for SSE event plumbing (events are fire-and-forget here).
- Requires the agent feature models to be configured in models.yaml.
- PipelineResult.rag_context is populated from the agent's chunk_registry.
- agent_meta and agent_findings are exposed on PipelineResult for inspection.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from uuid import UUID

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from src.db import get_session_factory
from src.eval.pipeline import PipelineResult, _run_answer, _run_direct_answer
from src.eval.schemas import EvalQuestion
from src.schemas.agent_findings import AgentFindings, AnalyticalFindings
from src.schemas.chat import ChatPipelineState
from src.schemas.query_router import ChatScope, RouterInput
from src.schemas.retrieval import RAGContext
from src.services.chat.agent_loop import AgentLoopMeta, _order_chunks, run_agent_loop
from src.services.chat.findings_processor import (
    ProcessedFindings,
    _render_findings_block,
    _render_observations_block,
    process_findings,
)
from src.services.llm_router import LLMRouter, get_router
from src.services.retrieval.context_assembler import assemble_rag_context
from src.services.retrieval.payload_hydrator import get_chunk_prompt_payloads
from src.services.retrieval.reranker import get_reranker
from src.services.router.router import route_query
from src.utils.config import get_agent_config, get_redis_app_url

logger = logging.getLogger(__name__)


@dataclass
class AgentPipelineResult(PipelineResult):
    """PipelineResult extended with agentic metadata."""

    agent_meta: AgentLoopMeta | None = None
    agent_findings: AgentFindings | AnalyticalFindings | None = None
    processed_findings: ProcessedFindings | None = None
    query_shape: str | None = None


def _make_redis() -> Redis:
    return Redis.from_url(get_redis_app_url(), decode_responses=True)


async def run_one(
    question: EvalQuestion,
    session: AsyncSession,
    user_id: UUID,
    model_id: str,
    prompt_version: str = "v3_agent_synthesis",
    reasoning_effort: str | None = None,
    max_tokens: int | None = None,
    verbosity: str | None = None,
    llm_router: LLMRouter | None = None,
    retrieval_only: bool = False,
    redis: Redis | None = None,
) -> AgentPipelineResult:
    """Run a single eval question through the agentic pipeline.

    Falls back to direct_answer (no retrieval) for non-retrieval routes, identical
    to the classic pipeline. For retrieval routes the agent loop is used.

    redis: pass an existing Redis client to reuse connections across calls; if None
           a new client is created and closed after each call.
    """
    router = llm_router or get_router()
    reranker = get_reranker()
    cfg = get_agent_config()

    router_inp = RouterInput(
        query=question.question,
        scope=ChatScope(mode="allDocs"),
    )
    router_out, scope_result = await route_query(
        router_inp,
        user_id=user_id,
        llm_router=router,
        session=session,
    )
    route = router_out.route
    query_shape = getattr(router_out, "query_shape", None)

    if route != "retrieval":
        if retrieval_only:
            return AgentPipelineResult(
                route=route,
                rag_context=None,
                retrieval_trace=None,
                answer=None,
                citation_spans=[],
                usage=None,
                query_shape=query_shape,
            )
        answer, spans, stats = await _run_direct_answer(
            question.question,
            model_id,
            router,
            prompt_version,
            reasoning_effort,
            max_tokens,
            verbosity=verbosity,
        )
        return AgentPipelineResult(
            route=route,
            rag_context=None,
            retrieval_trace=None,
            answer=answer,
            citation_spans=spans,
            usage=stats,
            query_shape=query_shape,
        )

    # Build a minimal ChatPipelineState for run_agent_loop
    request_id = str(uuid.uuid4())

    _owns_redis = redis is None
    _redis = redis or _make_redis()

    # Fake a user_id-bearing LLMRequest stub so agent_loop can access user_id
    _eval_user_id = user_id

    class _LLMRequestStub:
        id = None
        user_id = _eval_user_id
        conversation_id = None

    state = ChatPipelineState(
        request_id=request_id,
        redis_app=_redis,
        session=session,
        user_query_raw=question.question,
        router_output=router_out,
        scope_result=scope_result,
        context_messages=[],
    )
    state.llm_request = _LLMRequestStub()  # type: ignore[assignment]

    try:
        tool_model_id: str = cfg["tool_model"]
        tool_llm = router.get(tool_model_id)

        chunk_registry, agent_findings, agent_meta = await run_agent_loop(
            state, tool_llm, session, _redis, request_id, reranker, get_session_factory()
        )
    finally:
        if _owns_redis:
            await _redis.aclose()

    ordered = _order_chunks(chunk_registry)
    chunk_ids = [c.chunk_id for c in ordered]
    payloads = await get_chunk_prompt_payloads(session, chunk_ids)
    rag_context, _ = assemble_rag_context(ordered, payloads, assume_unique=True)

    processed_findings: ProcessedFindings | None = None
    rag_context_str: str

    if agent_findings is not None:
        processed_findings = await process_findings(
            agent_findings,
            requested_currency=getattr(router_out, "requested_currency", None),
        )

        # Mirror the prod path: map finding chunk UUIDs to the synthesis context's
        # S-labels so the synthesis model only ever sees citable excerpt IDs.
        chunk_id_to_ref = {str(item.chunk_id): item.ref_id for item in rag_context.items}
        if processed_findings.analytical_findings is not None:
            findings_block = _render_observations_block(
                processed_findings.analytical_findings, chunk_id_to_ref=chunk_id_to_ref
            )
        else:
            findings_block = _render_findings_block(
                processed_findings, chunk_id_to_ref=chunk_id_to_ref
            )

        rag_context_str = findings_block + "\n\n" + (rag_context.formatted_context or "")
    else:
        rag_context_str = rag_context.formatted_context or "(No document context.)"

    if retrieval_only:
        return AgentPipelineResult(
            route=route,
            rag_context=rag_context,
            retrieval_trace=None,
            answer=None,
            citation_spans=[],
            usage=None,
            agent_meta=agent_meta,
            agent_findings=agent_findings,
            processed_findings=processed_findings,
            query_shape=query_shape,
        )

    # Synthesise answer using the agent synthesis prompt (same model as classic eval)
    answer, spans, stats = await _run_answer(
        question.question,
        _rag_context_with_override(rag_context, rag_context_str),
        model_id,
        router,
        prompt_version,
        reasoning_effort,
        max_tokens,
        verbosity=verbosity,
    )

    return AgentPipelineResult(
        route=route,
        rag_context=rag_context,
        retrieval_trace=None,
        answer=answer,
        citation_spans=spans,
        usage=stats,
        agent_meta=agent_meta,
        agent_findings=agent_findings,
        processed_findings=processed_findings,
        query_shape=query_shape,
    )


def _rag_context_with_override(base: RAGContext, formatted_context: str) -> RAGContext:
    """Return a RAGContext whose formatted_context is replaced (items kept intact)."""
    return RAGContext(
        formatted_context=formatted_context,
        items=base.items,
        chunk_count=base.chunk_count,
    )
