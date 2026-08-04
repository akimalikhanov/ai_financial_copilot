"""The eval pipeline: drives run_agent the same way the Celery chat task does.

- Uses run_agent (multi-turn tool-calling + synthesis), matching what production serves.
- Requires a Redis connection for SSE event plumbing (events are fire-and-forget here).
- Requires the agent feature models to be configured in models.yaml.
- PipelineResult.rag_context is populated from the agent's synthesized context.
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
from src.eval.schemas import EvalQuestion
from src.schemas.agent_findings import AgentFindings, AnalyticalFindings
from src.schemas.chat import ChatPipelineState
from src.schemas.query_router import ChatScope, RouterInput
from src.schemas.retrieval import AnswerCitationSpan, RAGContext, RetrievalTrace
from src.services.chat.agent import run_agent
from src.services.chat.agent.processor import ProcessedFindings
from src.services.chat.agent.state import AgentLoopMeta, get_agent_settings
from src.services.chat.citation_parser import BracketCitationParser
from src.services.llm_adapters.base_adapter import ChatMessage, LLMResponseStats, Role
from src.services.llm_router import LLMRouter, get_router
from src.services.prompts.prompt_renderer import get_prompt_renderer, get_system_prompt
from src.services.retrieval.reranker import get_reranker
from src.services.router.router import route_query
from src.utils.config import get_redis_app_url

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    route: str
    rag_context: RAGContext | None
    retrieval_trace: RetrievalTrace | None
    answer: str | None
    citation_spans: list[AnswerCitationSpan]
    usage: LLMResponseStats | None
    excluded_reason: str | None = None


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
    settings = get_agent_settings()

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

    # Build a minimal ChatPipelineState for run_agent
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
        tool_model_id: str = settings.tool_model
        tool_llm = router.get(tool_model_id)

        agent_result = await run_agent(
            state, tool_llm, session, _redis, request_id, reranker, get_session_factory()
        )
    finally:
        if _owns_redis:
            await _redis.aclose()

    agent_meta = agent_result.meta

    if retrieval_only:
        return AgentPipelineResult(
            route=route,
            rag_context=agent_result.rag_context,
            retrieval_trace=None,
            answer=None,
            citation_spans=[],
            usage=None,
            agent_meta=agent_meta,
            agent_findings=agent_result.findings,
            processed_findings=agent_result.processed,
            query_shape=query_shape,
        )

    # Synthesise answer using the agent synthesis prompt (same model as classic eval)
    answer, spans, stats = await _run_answer(
        question.question,
        _rag_context_with_override(agent_result.rag_context, agent_result.synthesis_context),
        model_id,
        router,
        prompt_version,
        reasoning_effort,
        max_tokens,
        verbosity=verbosity,
    )

    return AgentPipelineResult(
        route=route,
        rag_context=agent_result.rag_context,
        retrieval_trace=None,
        answer=answer,
        citation_spans=spans,
        usage=stats,
        agent_meta=agent_meta,
        agent_findings=agent_result.findings,
        processed_findings=agent_result.processed,
        query_shape=query_shape,
    )


def _rag_context_with_override(base: RAGContext, formatted_context: str) -> RAGContext:
    """Return a RAGContext whose formatted_context is replaced (items kept intact)."""
    return RAGContext(
        formatted_context=formatted_context,
        items=base.items,
        chunk_count=base.chunk_count,
    )


async def _complete(
    context: str,
    question: str,
    model_id: str,
    router: LLMRouter,
    prompt_version: str,
    reasoning_effort: str | None,
    max_tokens: int | None,
    verbosity: str | None,
) -> tuple[str, BracketCitationParser, LLMResponseStats | None]:
    messages = [
        ChatMessage(role=Role.system, content=get_system_prompt(version=prompt_version)),
        ChatMessage(
            role=Role.user,
            content=get_prompt_renderer().render_user_message(
                context=context, user_query=question, version="v1"
            ),
        ),
    ]
    kwargs: dict = {}
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if verbosity:
        kwargs["verbosity"] = verbosity
    resp = await router.get(model_id).complete(messages=messages, temperature=0.0, **kwargs)
    parser = BracketCitationParser()
    out = parser.feed(resp.text or "")
    fin = parser.finalize()
    return out.visible_text + fin.visible_text, parser, resp.stats


async def _run_direct_answer(
    question: str,
    model_id: str,
    router: LLMRouter,
    prompt_version: str,
    reasoning_effort: str | None = None,
    max_tokens: int | None = None,
    verbosity: str | None = None,
) -> tuple[str, list[AnswerCitationSpan], LLMResponseStats | None]:
    answer, _parser, stats = await _complete(
        "", question, model_id, router, prompt_version, reasoning_effort, max_tokens, verbosity
    )
    return answer, [], stats


async def _run_answer(
    question: str,
    rag_context: RAGContext,
    model_id: str,
    router: LLMRouter,
    prompt_version: str,
    reasoning_effort: str | None = None,
    max_tokens: int | None = None,
    verbosity: str | None = None,
) -> tuple[str, list[AnswerCitationSpan], LLMResponseStats | None]:
    answer, parser, stats = await _complete(
        rag_context.formatted_context,
        question,
        model_id,
        router,
        prompt_version,
        reasoning_effort,
        max_tokens,
        verbosity,
    )
    return answer, list(parser.all_spans), stats
