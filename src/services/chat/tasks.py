"""Chat pipeline task."""

from __future__ import annotations

import asyncio
import contextlib
import json as _json
import logging
from time import perf_counter
from typing import Any
from uuid import UUID

from celery.signals import setup_logging, worker_process_init, worker_process_shutdown
from langfuse import propagate_attributes
from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from src.api.logging import configure_worker_logging, worker_request_context
from src.celery_app import celery_app
from src.models.message import Message, MessageStatus
from src.observability import langfuse as lf_client
from src.observability.metrics import (
    AGENT_ITERATIONS,
    GUARDRAIL_BLOCKS,
    LLM_CACHE_HIT_TOKENS,
    LLM_COST,
    LLM_TOKENS,
    PIPELINE_ERRORS,
    RAG_CITATIONS,
    RAG_CONTEXT_TOKENS,
    ROUTER_DECISIONS,
)
from src.redis_client import add_event
from src.repository import ConversationRepository, LLMRequestRepository, MessageRepository
from src.repository.llm_request_repository import stats_to_request_kwargs
from src.schemas import chat as schemas
from src.schemas.chat import ChatPipelineState
from src.schemas.query_router import ChatScope, DocumentScopeResult, RouterInput
from src.services.chat.agent import run_agent
from src.services.chat.agent.state import get_agent_settings
from src.services.chat.citation_parser import BracketCitationParser
from src.services.chat.confidence import compute_confidence, has_ungrounded_claims
from src.services.chat.events import (
    ThinkingStripper,
    build_all_references,
    build_references_list,
    build_usage_event,
    error_event,
    out_of_scope_response,
    span_to_dict,
)
from src.services.chat.naming import generate_conversation_title
from src.services.context import ConversationHistory, assemble_prompt
from src.services.llm_router import LLMRouter, get_router
from src.services.prompts.prompt_renderer import get_prompt_renderer, get_system_prompt
from src.services.retrieval.reranker import Reranker, get_reranker
from src.services.router.router import route_query
from src.services.security.injection_detector import InjectionSignal, scan_user_input
from src.utils.config import (
    get_conversation_naming_config,
    get_db_url,
    get_injection_scan_user_input_enabled,
    get_redis_app_url,
)

logger = logging.getLogger(__name__)

_STAGE_OBS_TYPES: dict[str, str] = {
    "route_query": "chain",
    "agent_loop": "chain",
}

_worker_loop: asyncio.AbstractEventLoop | None = None
_redis_app: Redis | None = None
_engine = None
_session_factory: async_sessionmaker[AsyncSession] | None = None
_router: LLMRouter | None = None
_reranker: Reranker | None = None


def _parse_scope(raw: object) -> ChatScope | None:
    """Parse scope dict from message metadata. Converts camelCase docIds → doc_ids."""
    if not isinstance(raw, dict):
        return None
    try:
        normalized = {
            "mode": raw.get("mode", "allDocs"),
            "doc_ids": [str(x) for x in raw.get("docIds", [])],
            "filters": raw.get("filters", {}),
        }
        return ChatScope.model_validate(normalized)
    except Exception:
        logger.warning("scope_parse_failed", extra={"raw": str(raw)[:200]})
        return None


_SCOPE_MODE_LABELS = {
    "allDocs": "All documents",
    "selectedDocs": "Selected documents",
    "thisDoc": "Single document",
    "filteredByMetadata": "Filtered by metadata",
}


def _scope_summary(
    chat_scope: ChatScope | None,
    scope_result: object,
) -> dict[str, object]:
    """Build a clear, human-readable scope summary for Langfuse traces.

    Combines what the user selected (mode + filters) with how it resolved
    (source, doc count, per-entity companies) into a flat, glanceable dict.
    """
    requested_mode = chat_scope.mode if chat_scope else "allDocs"
    summary: dict[str, object] = {
        "requested_mode": requested_mode,
        "requested_mode_label": _SCOPE_MODE_LABELS.get(requested_mode, requested_mode),
    }
    if chat_scope and chat_scope.mode == "filteredByMetadata":
        f = chat_scope.filters
        filters: dict[str, object] = {}
        if f.company:
            filters["company"] = f.company
        if f.year:
            filters["year"] = f.year
        if f.type:
            filters["type"] = f.type
        if filters:
            summary["requested_filters"] = filters
    elif chat_scope and chat_scope.mode in ("selectedDocs", "thisDoc"):
        summary["requested_doc_count"] = len(chat_scope.doc_ids)

    if isinstance(scope_result, DocumentScopeResult):
        doc_ids = scope_result.doc_ids
        companies = (
            sorted(scope_result.per_entity_doc_ids.keys())
            if scope_result.per_entity_doc_ids
            else []
        )
        summary["resolved_source"] = scope_result.source
        summary["resolved_doc_count"] = len(doc_ids) if doc_ids is not None else "all"
        summary["resolved_companies"] = companies
        # One-line headline so the scope is legible at a glance in the trace.
        count = summary["resolved_doc_count"]
        co = f" · {', '.join(companies)}" if companies else ""
        summary["headline"] = (
            f"{summary['requested_mode_label']} → {count} doc(s){co} [{scope_result.source}]"
        )
    else:
        summary["resolved_source"] = None
        summary["headline"] = f"{summary['requested_mode_label']} (no resolution)"
    return summary


def _initialize_worker_resources() -> None:
    global _worker_loop, _redis_app, _engine, _session_factory, _router, _reranker
    if _worker_loop is None or _worker_loop.is_closed():
        _worker_loop = asyncio.new_event_loop()
    if _redis_app is None:
        _redis_app = Redis.from_url(get_redis_app_url(), decode_responses=True)
    if _engine is None:
        _engine = create_async_engine(get_db_url(), poolclass=NullPool)
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            _engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
    if _router is None:
        _router = get_router()
    if _reranker is None:
        _reranker = get_reranker()
    lf_client.initialize()


def _get_worker_loop() -> asyncio.AbstractEventLoop:
    if _worker_loop is None or _worker_loop.is_closed():
        raise RuntimeError("Chat worker loop is not initialized")
    return _worker_loop


def _get_router() -> LLMRouter:
    if _router is None:
        raise RuntimeError("Chat worker LLM router is not initialized")
    return _router


def _get_reranker() -> Reranker:
    if _reranker is None:
        raise RuntimeError("Chat worker reranker is not initialized")
    return _reranker


def _get_redis_app() -> Redis:
    if _redis_app is None:
        raise RuntimeError("Chat worker Redis is not initialized")
    return _redis_app


def _get_session_factory() -> async_sessionmaker[AsyncSession]:
    if _session_factory is None:
        raise RuntimeError("Chat worker session factory is not initialized")
    return _session_factory


@setup_logging.connect
def _on_celery_setup_logging(**_kwargs: object) -> None:
    configure_worker_logging()


@worker_process_init.connect
def _on_worker_process_init(**_kwargs: object) -> None:
    global _worker_loop, _redis_app, _engine, _session_factory, _router, _reranker
    configure_worker_logging()
    _initialize_worker_resources()


@worker_process_shutdown.connect
def _on_worker_process_shutdown(**_kwargs: object) -> None:
    global _worker_loop, _redis_app, _engine, _session_factory, _router, _reranker
    if _worker_loop is None or _worker_loop.is_closed():
        return
    if _router is not None:
        _worker_loop.run_until_complete(_router.close())
    if _reranker is not None and hasattr(_reranker, "aclose"):
        _worker_loop.run_until_complete(_reranker.aclose())
    if _redis_app is not None:
        _worker_loop.run_until_complete(_redis_app.aclose())
    if _engine is not None:
        _worker_loop.run_until_complete(_engine.dispose())
    lf_client.flush()
    lf_client.reset()
    _router = None
    _reranker = None
    _redis_app = None
    _engine = None
    _session_factory = None
    _worker_loop.close()
    _worker_loop = None


async def _run_chat_pipeline(request_id: str) -> None:
    with worker_request_context(request_id):
        await _run_chat_pipeline_inner(request_id)


async def _run_chat_pipeline_inner(request_id: str) -> None:
    sf = _get_session_factory()
    redis_app = _get_redis_app()
    pipeline_started_at = perf_counter()
    stage_start = perf_counter()
    stage_times: dict[str, float] = {}
    agent_findings_json: str | None = None  # set by agent branch; used in persist
    # 7 stages, plus scan_user_input when the guardrail is on.
    stage_total = 7 + int(get_injection_scan_user_input_enabled())
    stage_index = 0
    current_stage = "initializing"

    async def _log_stage(stage_name: str, **extra_fields: Any) -> None:
        nonlocal stage_index, current_stage, stage_start, _stage_stack
        if current_stage != "initializing":
            stage_times[current_stage] = round(perf_counter() - stage_start, 3)
            _stage_stack.close()
        stage_index += 1
        current_stage = f"{stage_index:02d}_{stage_name}"
        stage_start = perf_counter()
        logger.info(
            f"pipeline.stage [{stage_index}/{stage_total}] {stage_name}",
            extra={"request_id": request_id, "stage": stage_name, **extra_fields},
        )
        await add_event(
            redis_app,
            request_id,
            "stage",
            {"stage": stage_name, "index": stage_index, "total": stage_total},
        )
        _stage_stack = contextlib.ExitStack()
        if lf:
            _stage_stack.enter_context(
                lf.start_as_current_observation(
                    as_type=_STAGE_OBS_TYPES.get(stage_name, "span"),  # type: ignore[arg-type]
                    name=stage_name,
                )
            )

    lf = lf_client.get_client()
    _lf_stack = contextlib.ExitStack()
    _stage_stack: contextlib.ExitStack = contextlib.ExitStack()
    _gen_stack: contextlib.ExitStack = contextlib.ExitStack()
    _gen: object = None
    _root_span: object = None
    if lf:
        _root_span = _lf_stack.enter_context(
            lf.start_as_current_observation(
                as_type="chain",
                name="chat_pipeline",
                trace_context={"trace_id": UUID(request_id).hex},
                input={"request_id": request_id},
            )
        )

    try:
        async with sf() as session:
            state = ChatPipelineState(
                request_id=request_id,
                redis_app=redis_app,
                session=session,
            )
            llm_request_repo = LLMRequestRepository(session)
            message_repo = MessageRepository(session)
            conversation_repo = ConversationRepository(session)

            # 1. load_and_validate_request
            await _log_stage("load_and_validate_request")
            llm_request = await llm_request_repo.get_by_id(UUID(request_id))
            if not llm_request:
                logger.error("llm_request_not_found", extra={"request_id": request_id})
                await add_event(
                    redis_app, request_id, "error", error_event(LookupError("Request not found"))
                )
                return

            state.llm_request = llm_request
            state.conversation_id = llm_request.conversation_id
            state.assistant_message_id = llm_request.assistant_message_id

            if not state.assistant_message_id:
                logger.error("no_assistant_placeholder", extra={"request_id": request_id})
                await add_event(
                    redis_app,
                    request_id,
                    "error",
                    error_event(ValueError("No assistant placeholder")),
                )
                return

            assistant_msg = await message_repo.get_by_id(state.assistant_message_id)
            state.assistant_seq = assistant_msg.seq if assistant_msg else 0
            await llm_request_repo.update_status(UUID(request_id), "streaming")

            if lf:
                _lf_stack.enter_context(
                    propagate_attributes(
                        user_id=str(llm_request.user_id) if llm_request.user_id else None,
                        session_id=str(state.conversation_id),
                        metadata={"request_id": request_id, "model": llm_request.model},
                    )
                )

            # 2. build_conversation_context
            await _log_stage("build_conversation_context")
            history = ConversationHistory(redis_app, message_repo)
            state.history = history
            state.context_messages = await history.load(
                state.conversation_id,
                before_seq=state.assistant_seq,
                snapshot_seq=llm_request.snapshot_seq or 0,
            )

            last_user = next(
                (m for m in reversed(state.context_messages) if m.role == schemas.Role.user),
                None,
            )
            state.user_query_raw = last_user.content if last_user else ""
            if lf:
                lf.update_current_span(
                    input={"request_id": request_id, "query": state.user_query_raw}
                )

            injection_signal: InjectionSignal | None = None
            # 2.5 scan_user_input (skipped when INJECTION_SCAN_USER_INPUT=false)
            if get_injection_scan_user_input_enabled():
                await _log_stage("scan_user_input")
                injection_signal = scan_user_input(state.user_query_raw)
                state.user_query_raw = injection_signal.sanitized_text

                logger.info(
                    "pipeline.injection_scan",
                    extra={
                        "request_id": request_id,
                        "injection_score": injection_signal.score,
                        "injection_severity": injection_signal.severity,
                        "matched_rules": injection_signal.matched_rules,
                        "stripped_chars": injection_signal.stripped_chars,
                    },
                )

                if lf:
                    lf_trace_id = UUID(request_id).hex
                    lf.create_score(
                        name="injection_score",
                        value=float(injection_signal.score),
                        trace_id=lf_trace_id,
                    )
                    lf.create_score(
                        name="injection_severity",
                        value=injection_signal.severity,
                        trace_id=lf_trace_id,
                    )

                if injection_signal.severity == "block":
                    GUARDRAIL_BLOCKS.labels("injection").inc()
                    refusal_text = (
                        "I'm sorry, but I can't process that request. "
                        "Please ask a financial question about your documents."
                    )
                    await add_event(redis_app, request_id, "delta", {"text": refusal_text})
                    await message_repo.update_on_final(
                        message_id=state.assistant_message_id,
                        content=refusal_text,
                        raw_content=refusal_text,
                        request_id=UUID(request_id),
                        trace={
                            "guardrails": {
                                "injection": {
                                    "score": injection_signal.score,
                                    "severity": injection_signal.severity,
                                    "matched_rules": injection_signal.matched_rules,
                                    "stripped_chars": injection_signal.stripped_chars,
                                }
                            }
                        },
                    )
                    await llm_request_repo.update_status(UUID(request_id), "completed")
                    await conversation_repo.update_on_message(
                        conversation_id=state.conversation_id,
                        message_id=state.assistant_message_id,
                        new_seq=state.assistant_seq,
                    )
                    usage_data = build_usage_event(
                        state.assistant_message_id,
                        state.assistant_seq,
                        None,
                    )
                    await add_event(redis_app, request_id, "usage", usage_data)
                    await session.commit()
                    try:
                        await state.history.append_assistant(
                            state.conversation_id,
                            refusal_text,
                            state.assistant_seq,
                        )
                    except Exception:
                        logger.warning("chat_tail_append_failed", extra={"request_id": request_id})
                    logger.info("pipeline.injection_blocked", extra={"request_id": request_id})
                    return

            # 3. route_query
            await _log_stage("route_query")
            router = _get_router()

            user_db_msg = (
                await message_repo.get_by_id(llm_request.user_message_id)
                if llm_request.user_message_id
                else None
            )
            raw_scope = (user_db_msg.message_metadata or {}).get("scope") if user_db_msg else None
            chat_scope = _parse_scope(raw_scope)

            router_input = RouterInput(
                query=state.user_query_raw,
                scope=chat_scope,
                conversation_history=[
                    {"role": m.role.value, "content": m.content}
                    for m in (state.context_messages or [])
                ],
            )
            state.router_output, state.scope_result = await route_query(
                router_input,
                user_id=llm_request.user_id,
                llm_router=router,
                session=session,
                parent_request_id=llm_request.id,
                conversation_id=state.conversation_id,
            )
            ROUTER_DECISIONS.labels(state.router_output.route).inc()
            _scope_doc_ids = (
                [str(d) for d in state.scope_result.doc_ids]
                if state.scope_result and state.scope_result.doc_ids is not None
                else None
            )
            _scope_per_entity = (
                {
                    entity: [str(d) for d in ids]
                    for entity, ids in state.scope_result.per_entity_doc_ids.items()
                }
                if state.scope_result and state.scope_result.per_entity_doc_ids
                else None
            )
            _scope_entity_manifest = (
                [item.model_dump() for item in state.scope_result.entity_manifest]
                if state.scope_result and state.scope_result.entity_manifest
                else None
            )
            logger.info(
                "rag_route",
                extra={
                    "request_id": request_id,
                    "route": state.router_output.route,
                    "entities": [e.model_dump() for e in state.router_output.entities],
                    "user_intent": state.router_output.user_intent,
                    "scope_source": state.scope_result.source if state.scope_result else None,
                    "scope_doc_ids": _scope_doc_ids,
                    "scope_per_entity_doc_ids": _scope_per_entity,
                },
            )
            if lf:
                scope_summary = _scope_summary(chat_scope, state.scope_result)
                lf.update_current_span(
                    input={"query": state.user_query_raw},
                    output={
                        "route": state.router_output.route,
                        "query_shape": getattr(state.router_output, "query_shape", None),
                        "entities": [e.model_dump() for e in state.router_output.entities],
                        "user_intent": state.router_output.user_intent,
                        "scope": scope_summary,
                    },
                    metadata={
                        "scope": scope_summary,
                        "scope_source": state.scope_result.source if state.scope_result else None,
                        "scope_doc_ids": _scope_doc_ids,
                        "scope_doc_count": len(_scope_doc_ids) if _scope_doc_ids is not None else 0,
                        "scope_per_entity_doc_ids": _scope_per_entity,
                        "scope_entity_manifest": _scope_entity_manifest,
                    },
                )
                # Surface scope at the trace root so it's visible without drilling into
                # the route stage. The root span owns trace-level IO in langfuse v3.
                if _root_span is not None:
                    with contextlib.suppress(Exception):
                        _root_span.set_trace_io(  # type: ignore[attr-defined]
                            output={"scope": scope_summary["headline"]}
                        )
                        _root_span.update(metadata={"scope": scope_summary})  # type: ignore[attr-defined]

            # Early-exit: out_of_scope — skip RAG + LLM, emit redirect and persist
            if state.router_output.route == "out_of_scope":
                redirect_text = out_of_scope_response()
                await add_event(redis_app, request_id, "delta", {"text": redirect_text})
                await message_repo.update_on_final(
                    message_id=state.assistant_message_id,
                    content=redirect_text,
                    raw_content=redirect_text,
                    request_id=UUID(request_id),
                )
                await llm_request_repo.update_status(UUID(request_id), "completed")
                await conversation_repo.update_on_message(
                    conversation_id=state.conversation_id,
                    message_id=state.assistant_message_id,
                    new_seq=state.assistant_seq,
                )
                usage_data = build_usage_event(
                    state.assistant_message_id,
                    state.assistant_seq,
                    None,
                )
                await add_event(redis_app, request_id, "usage", usage_data)
                await session.commit()
                try:
                    await state.history.append_assistant(
                        state.conversation_id,
                        redirect_text,
                        state.assistant_seq,
                    )
                except Exception:
                    logger.warning("chat_tail_append_failed", extra={"request_id": request_id})
                logger.info("pipeline.out_of_scope", extra={"request_id": request_id})
                return

            agent_settings = get_agent_settings()
            # `user_id` is nullable on LLMRequest, and the agent loop cannot search
            # without one — route that case to the no-context path explicitly.
            if state.router_output.route == "retrieval" and llm_request.user_id is not None:
                _tool_model_id: str = agent_settings.tool_model
                _tool_llm = router.get(_tool_model_id)
                if not _tool_llm.capabilities.get("tool_calling", False):
                    raise RuntimeError(
                        f"AGENT_TOOL_MODEL={_tool_model_id!r} does not have tool_calling: true in models.yaml"
                    )

                # Step 25: mark this request as agentic for DB queries/dashboards
                llm_request.request_type = "chat_agent"
                await session.flush()

                await _log_stage(
                    "agent_loop", model=_tool_llm.model_id, provider=_tool_llm.provider
                )

                _agent_lf_stack = contextlib.ExitStack()
                if lf:
                    _agent_lf_stack.enter_context(
                        lf.start_as_current_observation(
                            as_type="chain",
                            name="agent_loop",
                            input={
                                "query": state.user_query_raw,
                                "query_shape": getattr(state.router_output, "query_shape", None),
                                "tool_model": _tool_model_id,
                            },
                        )
                    )
                try:
                    agent_result = await run_agent(
                        state,
                        _tool_llm,
                        session,
                        redis_app,
                        request_id,
                        _get_reranker(),
                        _get_session_factory(),
                    )
                    agent_meta = agent_result.meta
                    if lf:
                        lf.update_current_span(
                            output={
                                "iterations": agent_meta.iterations,
                                "tool_calls_total": agent_meta.tool_calls_total,
                                "convergence_reason": agent_meta.convergence_reason,
                                "sealed": agent_meta.sealed,
                                "chunks_collected": len(agent_result.rag_context.items),
                            },
                            metadata={
                                "input_tokens_total": agent_meta.input_tokens_total,
                                "output_tokens_total": agent_meta.output_tokens_total,
                                "cost_usd_total": agent_meta.cost_usd_total,
                            },
                        )
                finally:
                    _agent_lf_stack.close()

                AGENT_ITERATIONS.observe(agent_meta.iterations)
                state.agent_meta = agent_meta

                state.rag_context = agent_result.rag_context
                state.rag_context_str = agent_result.synthesis_context

                if agent_result.findings is not None:
                    agent_findings_json = agent_result.findings.model_dump_json()
                if agent_result.processed is not None:
                    state.agent_answer_entity = agent_result.processed.answer_entity
                    state.agent_fx_rates = agent_result.processed.fx_rates_used
                    state.agent_currency_converted = agent_result.processed.currency_converted

                logger.info(
                    "agent_loop_complete",
                    extra={
                        "request_id": request_id,
                        "iterations": agent_meta.iterations,
                        "tool_calls_total": agent_meta.tool_calls_total,
                        "convergence_reason": agent_meta.convergence_reason,
                        "chunks_collected": len(agent_result.rag_context.items),
                        "findings_set": agent_result.findings is not None,
                        "sealed": agent_meta.sealed,
                    },
                )

            else:
                state.rag_context_str = "(No document context - general question.)"

            top_score = (
                state.rag_context.items[0].score
                if state.rag_context and state.rag_context.items
                else None
            )
            num_chunks = len(state.rag_context.items) if state.rag_context else 0

            # 5. render_prompt
            await _log_stage("render_prompt")
            # ~4 chars/token heuristic — a cheap, bounded proxy for synthesis context size.
            RAG_CONTEXT_TOKENS.observe(len(state.rag_context_str or "") / 4)
            try:
                llm = router.get(llm_request.model)
            except Exception as e:
                logger.exception("llm_router_error", extra={"request_id": request_id})
                await llm_request_repo.update_status(UUID(request_id), "failed")
                await add_event(redis_app, request_id, "error", error_event(e))
                await session.commit()
                return

            prompt_version = "v3_agent_synthesis"

            renderer = get_prompt_renderer()
            state.params = dict(llm_request.request_params or {})
            state.adapter_messages = assemble_prompt(
                history=state.context_messages,
                system_prompt=get_system_prompt(version=prompt_version),
                rag_context=state.rag_context_str,
                user_query=state.user_query_raw,
                renderer=renderer,
            )
            if lf:
                lf.update_current_span(
                    input={
                        "prompt_version": prompt_version,
                        "num_chunks": num_chunks,
                        "context_messages": len(state.context_messages),
                    },
                    output={
                        "num_messages": len(state.adapter_messages),
                        "system_prompt_chars": len(state.adapter_messages[0].content or "")
                        if state.adapter_messages
                        else 0,
                        "rag_context_chars": len(state.rag_context_str or ""),
                    },
                )

            # 6. stream_llm_response
            await _log_stage("stream_llm_response", model=llm.model_id, provider=llm.provider)
            if lf:
                _gen = _gen_stack.enter_context(
                    lf.start_as_current_observation(
                        as_type="generation",
                        name="chat_model",
                        model=llm_request.model,
                        input=[
                            {"role": m.role.value, "content": m.content}
                            for m in state.adapter_messages
                        ],
                    )
                )
            temperature = state.params.get("temperature")
            max_tokens = state.params.get("max_tokens")
            extra = {
                k: v for k, v in state.params.items() if k not in ("temperature", "max_tokens")
            }

            stream = llm.stream(
                messages=state.adapter_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **extra,
            )

            parser = BracketCitationParser()
            think_stripper = ThinkingStripper()

            try:
                chunk = None
                async for chunk in stream:
                    state.accumulated_content += chunk.text  # raw for DB
                    # Strip [S1] markers, track spans. Emitted for every chunk, including
                    # the final one — a final chunk can carry text, and skipping it would
                    # drop that text and any spans it completed from the SSE stream.
                    result = parser.feed(think_stripper.feed(chunk.text))
                    state.clean_content += result.visible_text
                    if result.visible_text:
                        await add_event(
                            redis_app, request_id, "delta", {"text": result.visible_text}
                        )
                    for span in result.completed_spans:
                        await add_event(redis_app, request_id, "citation_span", span_to_dict(span))

                if chunk is not None:
                    final_result = parser.finalize()
                    state.clean_content += final_result.visible_text
                    if final_result.visible_text:
                        await add_event(
                            redis_app, request_id, "delta", {"text": final_result.visible_text}
                        )
                    for span in final_result.completed_spans:
                        await add_event(redis_app, request_id, "citation_span", span_to_dict(span))

                    # Cited sources, in order of first appearance; all retrieved sources
                    # as a fallback when the model emitted no citations at all (e.g. a
                    # bare-number answer) so the evidence panel still populates. Built
                    # once here and shared by the SSE event, persistence, and usage.
                    ref_items: list[dict] = []
                    if state.rag_context:
                        cited_ref_ids: list[str] = []
                        for span in parser.all_spans:
                            for ref_id in span.ref_ids:
                                if ref_id not in cited_ref_ids:
                                    cited_ref_ids.append(ref_id)
                        ref_items = (
                            build_references_list(state.rag_context, cited_ref_ids)
                            if cited_ref_ids
                            else build_all_references(state.rag_context)
                        )
                    if ref_items:
                        await add_event(redis_app, request_id, "references", {"items": ref_items})

                    # 7. persist_and_emit
                    await _log_stage("persist_and_emit")
                    # Build citation metadata for persistence
                    citation_meta: dict = {}
                    if parser.all_spans:
                        citation_meta["citation_spans"] = [
                            span_to_dict(sp) for sp in parser.all_spans
                        ]
                    if ref_items:
                        citation_meta["references"] = ref_items

                    if state.rag_context and state.rag_context.items:
                        citation_meta["retrieved_chunks"] = [
                            {"chunk_id": str(item.chunk_id), "score": item.score}
                            for item in state.rag_context.items
                        ]

                    if agent_findings_json is not None:
                        citation_meta["agent_findings"] = agent_findings_json
                        # Persist the sealed/degraded marker beside the findings so a later
                        # turn's carry-over can tell a committed finalizer from a degraded
                        # partial serve (step 11 M2-synth) — `meta.sealed` is in-run only.
                        if state.agent_meta is not None:
                            citation_meta["agent_findings_sealed"] = state.agent_meta.sealed

                    # Finalize stage times (stream_llm_response ends here)
                    stage_times[current_stage] = round(perf_counter() - stage_start, 3)
                    total_time = round(perf_counter() - pipeline_started_at, 3)

                    confidence = compute_confidence(top_score, num_chunks)
                    ungrounded = has_ungrounded_claims(state.clean_content, parser.all_spans)

                    # Build pipeline trace
                    trace_payload: dict = {
                        "v": 1,
                        "stage_times": stage_times,
                        "total_time": total_time,
                        "router": {
                            "decision": state.router_output.route,
                            "reasoning": state.router_output.reasoning[:500]
                            if state.router_output.reasoning
                            else None,
                            "user_intent": state.router_output.user_intent,
                            "entities": [e.model_dump() for e in state.router_output.entities],
                            "scope_source": state.scope_result.source
                            if state.scope_result
                            else None,
                        },
                    }
                    if state.agent_meta is not None:
                        m = state.agent_meta
                        trace_payload["agent"] = {
                            "iterations": m.iterations,
                            "tool_calls_total": m.tool_calls_total,
                            "convergence_reason": m.convergence_reason,
                            "currency_normalized": state.agent_currency_converted,
                            "answer_entity": state.agent_answer_entity,
                            "fx_rates_used": state.agent_fx_rates,
                        }
                    trace_payload["guardrails"] = {
                        "confidence": confidence,
                        "top_reranker_score": top_score,
                        "num_chunks": num_chunks,
                        "ungrounded_claims": ungrounded,
                        **(
                            {
                                "injection": {
                                    "score": injection_signal.score,
                                    "severity": injection_signal.severity,
                                    "matched_rules": injection_signal.matched_rules,
                                    "stripped_chars": injection_signal.stripped_chars,
                                }
                            }
                            if injection_signal is not None
                            else {}
                        ),
                    }

                    lf_trace_id = UUID(request_id).hex if lf else None
                    await message_repo.update_on_final(
                        message_id=state.assistant_message_id,
                        content=state.clean_content,
                        raw_content=state.accumulated_content,
                        request_id=UUID(request_id),
                        metadata_updates=citation_meta or None,
                        trace=trace_payload,
                        trace_id=lf_trace_id,
                        agent_findings=_json.loads(agent_findings_json)
                        if agent_findings_json
                        else None,
                    )
                    RAG_CITATIONS.observe(len(parser.all_spans))

                    if chunk.stats:
                        _model = llm_request.model
                        if chunk.stats.input_tokens:
                            LLM_TOKENS.labels("input", _model).inc(chunk.stats.input_tokens)
                        if chunk.stats.output_tokens:
                            LLM_TOKENS.labels("output", _model).inc(chunk.stats.output_tokens)
                        if chunk.stats.cached_input_tokens:
                            LLM_CACHE_HIT_TOKENS.labels(_model).inc(chunk.stats.cached_input_tokens)
                        if chunk.stats.cost_usd:
                            LLM_COST.labels(_model).inc(chunk.stats.cost_usd)
                        await llm_request_repo.update_on_final(
                            request_id=UUID(request_id),
                            **stats_to_request_kwargs(chunk.stats),
                            trace_id=lf_trace_id,
                        )
                        if _gen is not None:
                            s = chunk.stats
                            _gen.update(  # type: ignore[union-attr]
                                output=state.accumulated_content,
                                usage_details={
                                    k: v
                                    for k, v in {
                                        "input": s.input_tokens,
                                        "output": s.output_tokens,
                                        "cache_read_input_tokens": s.cached_input_tokens,
                                        "total": s.total_tokens,
                                    }.items()
                                    if v is not None
                                },
                                cost_details={"total": s.cost_usd}
                                if s.cost_usd is not None
                                else None,
                            )
                    # Close chat_model generation while still inside stream_llm_response's
                    # contextvar scope — prevents contextvar corruption when persist_and_emit
                    # stage span was opened by _log_stage (which reset stream's token).
                    _gen_stack.close()
                    await llm_request_repo.update_status(UUID(request_id), "completed")
                    await conversation_repo.update_on_message(
                        conversation_id=state.conversation_id,
                        message_id=state.assistant_message_id,
                        new_seq=state.assistant_seq,
                    )

                    await add_event(
                        redis_app,
                        request_id,
                        "metadata",
                        {
                            "confidence": confidence,
                            "ungrounded_claims": ungrounded,
                            "route": state.router_output.route if state.router_output else None,
                        },
                    )
                    logger.info(
                        "confidence_score",
                        extra={
                            "request_id": request_id,
                            "confidence": confidence,
                            "top_score": top_score,
                            "num_chunks": num_chunks,
                            "ungrounded_claims": ungrounded,
                        },
                    )

                    usage_data = build_usage_event(
                        state.assistant_message_id,
                        state.assistant_seq,
                        chunk.stats,
                        citation_spans=parser.all_spans,
                        references=ref_items,
                    )
                    await session.commit()

                    try:
                        await state.history.append_assistant(
                            state.conversation_id,
                            state.clean_content,
                            state.assistant_seq,
                        )
                    except Exception:
                        logger.warning("chat_tail_append_failed", extra={"request_id": request_id})

                    # Auto-name conversation on the first exchange (seq 2 = first assistant reply)
                    # Must emit conversation_title BEFORE the usage event, since the frontend
                    # stops reading the stream as soon as it receives usage (the final sentinel).
                    naming_cfg = get_conversation_naming_config()
                    if naming_cfg["enabled"] and state.assistant_seq == 2 and state.user_query_raw:
                        try:
                            # Use a fresh session so the naming sub-request + title update
                            # commit together, independent of the main pipeline session.
                            async with sf() as naming_session:
                                title = await generate_conversation_title(
                                    query=state.user_query_raw,
                                    llm_router=router,
                                    model=naming_cfg["model"],
                                    max_len=naming_cfg["max_len"],
                                    session=naming_session,
                                    parent_request_id=UUID(request_id),
                                    conversation_id=state.conversation_id,
                                    user_id=llm_request.user_id,
                                )
                                if title:
                                    await ConversationRepository(naming_session).update(
                                        state.conversation_id, title=title
                                    )
                                await naming_session.commit()
                            if title:
                                await add_event(
                                    redis_app,
                                    request_id,
                                    "conversation_title",
                                    {"title": title, "conversation_id": str(state.conversation_id)},
                                )
                                logger.info(
                                    "conversation_named",
                                    extra={
                                        "request_id": request_id,
                                        "conversation_id": str(state.conversation_id),
                                        "title": title,
                                    },
                                )
                        except Exception:
                            logger.warning(
                                "conversation_naming_error", extra={"request_id": request_id}
                            )

                    await add_event(redis_app, request_id, "usage", usage_data)

            except Exception as e:
                logger.exception("llm_stream_error", extra={"request_id": request_id})
                await llm_request_repo.update_status(UUID(request_id), "failed")
                await llm_request_repo.update_on_final(
                    request_id=UUID(request_id),
                    error_code=type(e).__name__,
                    error_message=str(e),
                )
                result = await session.execute(
                    select(Message).where(Message.id == state.assistant_message_id)
                )
                msg = result.scalar_one_or_none()
                if msg:
                    msg.status = MessageStatus.error
                await add_event(redis_app, request_id, "error", error_event(e))
                await session.commit()

        logger.info(
            "pipeline.complete",
            extra={
                "request_id": request_id,
                "stage_times": stage_times,
                "total_time": round(perf_counter() - pipeline_started_at, 3),
            },
        )

    except Exception as exc:
        # current_stage is "NN_stage_name" (or "initializing") — strip the ordinal
        # prefix so the label stays stable across stage reordering.
        PIPELINE_ERRORS.labels(current_stage.split("_", 1)[-1]).inc()
        logger.exception(
            "pipeline.failed_at_stage",
            extra={"request_id": request_id, "stage": current_stage},
        )
        try:
            async with sf() as session:
                llm_repo = LLMRequestRepository(session)
                await llm_repo.update_status(UUID(request_id), "failed")
                await session.commit()
        except Exception:
            logger.exception("pipeline.set_failed_error", extra={"request_id": request_id})
        await add_event(redis_app, request_id, "error", error_event(exc))
        raise
    finally:
        _stage_stack.close()
        _gen_stack.close()
        _lf_stack.close()


@celery_app.task(bind=True, name="process_chat", acks_late=True, reject_on_worker_lost=True)
def process_chat(_self, request_id: str) -> None:
    """Celery task: process chat request."""
    _initialize_worker_resources()
    loop = _get_worker_loop()
    try:
        loop.run_until_complete(_run_chat_pipeline(request_id))
    finally:
        lf_client.flush()
