from __future__ import annotations

import logging
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from src.eval.schemas import EvalQuestion
from src.schemas.query_router import ChatScope, RouterInput
from src.schemas.query_transform import ScopeDocSummary
from src.schemas.retrieval import AnswerCitationSpan, RAGContext, RetrievalTrace
from src.services.chat.citation_parser import BracketCitationParser
from src.services.llm_adapters.base_adapter import ChatMessage, LLMResponseStats, Role
from src.services.llm_router import LLMRouter, get_router
from src.services.prompts.prompt_renderer import get_prompt_renderer, get_system_prompt
from src.services.retrieval.chat_rag import run_chat_rag_pipeline
from src.services.retrieval.query_transformer import rewrite_query
from src.services.retrieval.reranker import get_reranker
from src.services.router.router import route_query
from src.utils.config import get_chat_retrieval_timeout

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


async def run_one(
    question: EvalQuestion,
    session: AsyncSession,
    user_id: UUID,
    model_id: str,
    prompt_version: str = "v3_bracket",
    reasoning_effort: str | None = None,
    verbosity: str | None = None,
    llm_router: LLMRouter | None = None,
    retrieval_only: bool = False,
) -> PipelineResult:
    router = llm_router or get_router()

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

    if route != "retrieval":
        if retrieval_only:
            return PipelineResult(
                route=route,
                rag_context=None,
                retrieval_trace=None,
                answer=None,
                citation_spans=[],
                usage=None,
            )
        # direct_answer: run LLM for correctness, no retrieval
        answer, spans, stats = await _run_direct_answer(
            question.question,
            model_id,
            router,
            prompt_version,
            reasoning_effort,
            verbosity=verbosity,
        )
        return PipelineResult(
            route=route,
            rag_context=None,
            retrieval_trace=None,
            answer=answer,
            citation_spans=spans,
            usage=stats,
        )

    doc_ids = scope_result.doc_ids if scope_result else None
    scope_docs: list[ScopeDocSummary] = []

    transformed, _ = await rewrite_query(
        question.question,
        user_intent=router_out.user_intent,
        scope_docs=scope_docs,
        llm_router=router,
    )

    rag_context, retrieval_trace, _ = await run_chat_rag_pipeline(
        session,
        transformed=transformed,
        user_id=user_id,
        doc_ids=doc_ids,
        timeout=get_chat_retrieval_timeout(),
        reranker=get_reranker(),
    )

    if retrieval_only:
        return PipelineResult(
            route=route,
            rag_context=rag_context,
            retrieval_trace=retrieval_trace,
            answer=None,
            citation_spans=[],
            usage=None,
        )

    answer, spans, stats = await _run_answer(
        question.question,
        rag_context,
        model_id,
        router,
        prompt_version,
        reasoning_effort,
        verbosity=verbosity,
    )
    return PipelineResult(
        route=route,
        rag_context=rag_context,
        retrieval_trace=retrieval_trace,
        answer=answer,
        citation_spans=spans,
        usage=stats,
    )


async def _run_direct_answer(
    question: str,
    model_id: str,
    router: LLMRouter,
    prompt_version: str = "v3_bracket",
    reasoning_effort: str | None = None,
    max_tokens: int | None = None,
    verbosity: str | None = None,
) -> tuple[str, list[AnswerCitationSpan], LLMResponseStats | None]:
    system = get_system_prompt(version=prompt_version)
    renderer = get_prompt_renderer()
    user_msg = renderer.render_user_message(context="", user_query=question, version="v1")
    messages = [
        ChatMessage(role=Role.system, content=system),
        ChatMessage(role=Role.user, content=user_msg),
    ]
    llm = router.get(model_id)
    kwargs: dict = {}
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if verbosity:
        kwargs["verbosity"] = verbosity
    resp = await llm.complete(messages=messages, temperature=0.0, **kwargs)
    parser = BracketCitationParser()
    out = parser.feed(resp.text or "")
    fin = parser.finalize()
    answer = out.visible_text + fin.visible_text
    return answer, [], resp.stats


async def _run_answer(
    question: str,
    rag_context: RAGContext,
    model_id: str,
    router: LLMRouter,
    prompt_version: str = "v3_bracket",
    reasoning_effort: str | None = None,
    max_tokens: int | None = None,
    verbosity: str | None = None,
) -> tuple[str, list[AnswerCitationSpan], LLMResponseStats | None]:
    system = get_system_prompt(version=prompt_version)
    renderer = get_prompt_renderer()
    user_msg = renderer.render_user_message(
        context=rag_context.formatted_context,
        user_query=question,
        version="v1",
    )
    messages = [
        ChatMessage(role=Role.system, content=system),
        ChatMessage(role=Role.user, content=user_msg),
    ]
    llm = router.get(model_id)
    kwargs: dict = {}
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if verbosity:
        kwargs["verbosity"] = verbosity
    resp = await llm.complete(messages=messages, temperature=0.0, **kwargs)
    parser = BracketCitationParser()
    out = parser.feed(resp.text or "")
    fin = parser.finalize()
    answer = out.visible_text + fin.visible_text
    return answer, list(parser._spans), resp.stats
