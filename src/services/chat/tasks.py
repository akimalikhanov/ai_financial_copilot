"""Chat pipeline task."""

from __future__ import annotations

import asyncio
import contextlib
import json as _json
import logging
import os
from datetime import UTC, datetime
from time import perf_counter
from typing import Any, NamedTuple, get_args
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
from src.observability.langfuse import describe_error
from src.observability.langfuse import mark as lf_mark
from src.observability.langfuse import mark_current as lf_mark_current
from src.observability.metrics import (
    AGENT_ITERATIONS,
    AGENT_STOP_REASONS,
    AGENT_TOOL_ARG_ERRORS,
    CHAT_QUEUE_WAIT,
    CHAT_STAGE_DURATION,
    CHAT_TTFT,
    FOLLOWUP_DIRECT_ANSWER,
    FOLLOWUP_FINDINGS_CARRIED,
    GUARDRAIL_BLOCKS,
    LLM_CACHE_HIT_TOKENS,
    LLM_COST,
    LLM_TOKENS,
    PIPELINE_ERRORS,
    RAG_CITATIONS,
    RAG_CONTEXT_TOKENS,
    ROUTER_DECISIONS,
    observe_llm_latency,
)
from src.observability.trace_payload import cap_list, trace_params
from src.observability.trace_payload import dedup_scope as trace_dedup_scope
from src.redis_client import add_event, events_stream_key, expire_event_stream, get_activity_log
from src.repository import ConversationRepository, LLMRequestRepository, MessageRepository
from src.repository.llm_request_repository import stats_to_request_kwargs
from src.schemas import chat as schemas
from src.schemas.chat import ChatPipelineState
from src.schemas.query_router import ChatScope, DocumentScopeResult, RouterInput, RouterOutput
from src.services.chat.agent import run_agent
from src.services.chat.agent import tools as agent_tools
from src.services.chat.agent.loop import tool_model_chain
from src.services.chat.agent.state import ConvergenceReason, get_agent_settings, shape_config
from src.services.chat.citation_parser import BracketCitationParser
from src.services.chat.confidence import compute_confidence, uncited_fact_share
from src.services.chat.events import (
    ThinkingStripper,
    build_activity_event,
    build_all_references,
    build_references_list,
    build_scope_clarification_event,
    build_usage_event,
    clarification_text,
    error_event,
    out_of_scope_response,
    span_to_dict,
    too_broad_response,
)
from src.services.chat.naming import generate_conversation_title
from src.services.context import ConversationHistory, assemble_prompt
from src.services.context.turns import prior_turns
from src.services.llm_router import (
    FallbackStream,
    LLMRouter,
    get_router,
    trace_input,
    trace_usage,
)
from src.services.prompts.prompt_renderer import get_prompt_renderer, get_system_prompt
from src.services.retrieval.reranker import Reranker, get_reranker
from src.services.router.router import route_query
from src.services.router.scope_resolver import scope_outcome
from src.services.security.injection_detector import InjectionSignal, scan_user_input
from src.utils.config import (
    get_chat_max_request_age_seconds,
    get_conversation_naming_config,
    get_db_url,
    get_followup_max_inherit_hops,
    get_injection_scan_user_input_enabled,
    get_query_router_prompt_version,
    get_redis_app_url,
    get_scope_max_companies,
)

logger = logging.getLogger(__name__)

FINDINGS_BLOCK_MAX_CHARS = 20_000

SYNTHESIS_PROMPT_VERSION = "v5_agent_synthesis"

# Ceiling on acks_late redeliveries of one chat task, mirroring INGEST_MAX_ATTEMPTS. Past it
# the request is failed rather than retried, so a task that reliably kills its worker cannot
# loop forever re-billing the provider.
CHAT_MAX_ATTEMPTS = int(os.getenv("CHAT_MAX_ATTEMPTS", "2"))

# Ceiling on how stale a redelivered chat task may be and still be worth running. Bounds the
# hour that the broker's visibility timeout otherwise allows; see the guard in `process_chat`.
CHAT_MAX_REQUEST_AGE_SECONDS = get_chat_max_request_age_seconds()


def _observe_ttft(enqueued_at: datetime, query_shape: str) -> float:
    """Record the user-perceived time to first token and return it in seconds."""
    ttft = (datetime.now(UTC) - enqueued_at).total_seconds()
    CHAT_TTFT.labels(query_shape).observe(ttft)
    return ttft


_QUERY_SHAPES = ("extraction", "comparison", "analytical")


def _init_metric_series() -> None:
    """Create the labelled series at zero in this worker process.

    rate() never counts a series' first sample as an increase, so a series born at 1
    reads as zero rate and low-traffic quantiles come out NaN. Must run after fork:
    multiprocess values are per-PID.
    """
    for shape in (*_QUERY_SHAPES, "direct"):
        CHAT_TTFT.labels(shape)
    for reason in get_args(ConvergenceReason):
        for shape in (*_QUERY_SHAPES, "none"):
            AGENT_STOP_REASONS.labels(reason, shape)
    for tool in ("search_documents", agent_tools.REPORT_TOOL_NAME):
        AGENT_TOOL_ARG_ERRORS.labels(tool)


# Agent stops caused by a fault or a limit rather than the model finishing — marked WARNING
# on the agent_loop span so they can be filtered in Langfuse.
_AGENT_FAILED_STOPS: frozenset[ConvergenceReason] = frozenset(
    {"timeout", "deadline", "llm_error", "search_unavailable", "truncated"}
)

_STAGE_OBS_TYPES: dict[str, str] = {
    "route_query": "chain",
    "agent_loop": "chain",
}

_STAGE_LABELS: dict[str, str] = {
    "load_and_validate_request": "Loading request",
    "build_conversation_context": "Building context",
    "scan_user_input": "Scanning input",
    "route_query": "Routing query",
    "agent_loop": "Searching documents",
    "render_prompt": "Rendering prompt",
    "stream_llm_response": "Generating answer",
    "persist_and_emit": "Finalizing",
}

_worker_loop: asyncio.AbstractEventLoop | None = None
_redis_app: Redis | None = None
_engine = None
_session_factory: async_sessionmaker[AsyncSession] | None = None
_router: LLMRouter | None = None
_reranker: Reranker | None = None


class _CarriedFindings(NamedTuple):
    block: str | None
    hops: int
    doc_ids: list[str] | None
    outcome: str  # carried | none | dropped_hop_cap | dropped_scope


def _latest_findings_block(
    messages: list[schemas.ChatMessage] | None,
    current_doc_ids: list[str] | None = None,
    *,
    check_scope: bool = False,
) -> _CarriedFindings:
    """Carried findings block, the hop count it would have if inherited now, and why.

    The block is dropped past `FOLLOWUP_MAX_INHERIT_HOPS`, or when the resolved document
    scope has moved since it was produced — findings describing documents the user is no
    longer asking about are worse than a re-retrieval. A drop leaves the router with no
    carried data, so the follow-up re-retrieves.

    `check_scope` is off for the router call, which runs before scope is resolved.
    """
    for m in reversed(messages or []):
        if m.role == schemas.Role.assistant and m.findings_block:
            next_hops = m.findings_block_hops + 1
            if next_hops > get_followup_max_inherit_hops():
                return _CarriedFindings(None, 0, None, "dropped_hop_cap")
            if check_scope and _scope_moved(m.findings_block_doc_ids, current_doc_ids):
                return _CarriedFindings(None, 0, None, "dropped_scope")
            return _CarriedFindings(
                m.findings_block, next_hops, m.findings_block_doc_ids, "carried"
            )
    return _CarriedFindings(None, 0, None, "none")


def _turn_summary(
    router_output: RouterOutput | None, scope: DocumentScopeResult | None
) -> schemas.TurnSummary | None:
    """This turn's line in later turns' session index: what the router resolved."""
    if router_output is None:
        return None
    if scope is not None and scope.per_entity_doc_ids:
        entities = sorted(scope.per_entity_doc_ids)
    else:
        entities = [e.name for e in router_output.entities]
    return schemas.TurnSummary(
        route=router_output.route,
        query_shape=router_output.query_shape,
        entities=entities,
        doc_count=len(scope.doc_ids) if scope is not None and scope.doc_ids is not None else None,
    )


def _scope_moved(before: list[str] | None, now: list[str] | None) -> bool:
    """Whether the resolved document scope changed. None means "all documents", so it
    compares equal only to itself — a narrowing from all-docs is a real change."""
    if before is None or now is None:
        return not (before is None and now is None)
    return set(before) != set(now)


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
    _init_metric_series()


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
    current_stage = "initializing"
    current_stage_event_id: str | None = None

    async def _log_stage(stage_name: str, **extra_fields: Any) -> None:
        nonlocal current_stage, current_stage_event_id, stage_start, _stage_stack
        if current_stage != "initializing":
            elapsed = perf_counter() - stage_start
            stage_times[current_stage] = round(elapsed, 3)
            CHAT_STAGE_DURATION.labels(current_stage).observe(elapsed)
            _stage_stack.close()
            _, end_data = build_activity_event("stage_ended", event_id=current_stage_event_id)
            await add_event(redis_app, request_id, "activity", end_data)
        current_stage = stage_name
        stage_start = perf_counter()
        logger.info(
            f"pipeline.stage {stage_name}",
            extra={"request_id": request_id, "stage": stage_name, **extra_fields},
        )
        current_stage_event_id, start_data = build_activity_event(
            "stage_started", label=_STAGE_LABELS.get(stage_name, stage_name.replace("_", " "))
        )
        await add_event(redis_app, request_id, "activity", start_data)
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
        # Every generation in this trace logs a repeated message once (the agent resends
        # its whole transcript each turn).
        _lf_stack.enter_context(trace_dedup_scope())

    def _trace_io(**io: Any) -> None:
        """Trace-level input/output — what the Langfuse trace list shows per row."""
        if _root_span is not None:
            with contextlib.suppress(Exception):
                _root_span.set_trace_io(**io)  # type: ignore[attr-defined]

    def _score(name: str, value: float | str, data_type: str) -> None:
        if not lf:
            return
        trace_id = UUID(request_id).hex
        with contextlib.suppress(Exception):
            if isinstance(value, str):
                lf.create_score(name=name, value=value, data_type="CATEGORICAL", trace_id=trace_id)
            else:
                lf.create_score(
                    name=name,
                    value=value,
                    data_type=data_type,  # type: ignore[arg-type]
                    trace_id=trace_id,
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

            # acks_late + reject_on_worker_lost means a SIGKILLed task is redelivered. Without
            # this guard the redelivery re-runs the whole agent loop and synthesis, re-billing
            # the provider and appending a second answer to the same SSE stream. Must stay the
            # first thing after the load, ahead of any write or metric.
            if llm_request.status == "completed":
                logger.info("pipeline.already_completed", extra={"request_id": request_id})
                return

            # The companion to the guard above, for the case it cannot see. acks_late
            # redelivery is gated on the broker's visibility timeout, which is sized for
            # ingestion's 2700s parses and so runs an hour on the chat queue too (both queues
            # share one transport, and kombu restores from a single global unacked index). By
            # the time a redelivery of that age lands, the subscriber has already reported the
            # worker gone, so re-running the agent loop bills the provider for an answer
            # nobody is waiting for. Attempt count alone cannot catch this: a task killed on
            # its FIRST delivery arrives here at attempt 1, looking new.
            age_s = (datetime.now(UTC) - llm_request.created_at).total_seconds()
            if age_s > CHAT_MAX_REQUEST_AGE_SECONDS:
                await llm_request_repo.update_status(UUID(request_id), "failed")
                # Same reason as the max-attempts path below: commit before returning, or the
                # next redelivery reads the old state and takes this branch again.
                await session.commit()
                logger.warning(
                    "pipeline.request_too_stale",
                    extra={
                        "request_id": request_id,
                        "age_seconds": round(age_s, 1),
                        "max_age_seconds": CHAT_MAX_REQUEST_AGE_SECONDS,
                    },
                )
                await add_event(
                    redis_app,
                    request_id,
                    "error",
                    error_event(RuntimeError("Request expired before it could be processed")),
                )
                return

            attempt = await llm_request_repo.increment_attempt_count(UUID(request_id))
            if attempt > CHAT_MAX_ATTEMPTS:
                await llm_request_repo.update_status(UUID(request_id), "failed")
                # Committed before returning: an uncommitted status leaves the next
                # redelivery seeing the same state and looping forever.
                await session.commit()
                logger.warning(
                    "pipeline.max_attempts_exceeded",
                    extra={
                        "request_id": request_id,
                        "attempt": attempt,
                        "max_attempts": CHAT_MAX_ATTEMPTS,
                    },
                )
                await add_event(
                    redis_app,
                    request_id,
                    "error",
                    error_event(
                        RuntimeError(f"Exceeded max processing attempts ({CHAT_MAX_ATTEMPTS})")
                    ),
                )
                return

            if attempt > 1:
                # A redelivery must not append to the previous attempt's partial output —
                # a client reconnecting with Last-Event-ID would read two answers spliced
                # together. Start the stream clean.
                logger.info(
                    "pipeline.retry_attempt",
                    extra={"request_id": request_id, "attempt": attempt},
                )
                with contextlib.suppress(Exception):
                    await redis_app.delete(events_stream_key(request_id))

            CHAT_QUEUE_WAIT.observe((datetime.now(UTC) - llm_request.created_at).total_seconds())

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
            # Set by the API from the request's `allow_clarification`.
            allow_clarification = bool(
                (assistant_msg.message_metadata if assistant_msg else {}).get(
                    "allow_clarification", True
                )
            )
            await llm_request_repo.update_status(UUID(request_id), "streaming")

            if lf:
                _lf_stack.enter_context(
                    propagate_attributes(
                        user_id=str(llm_request.user_id) if llm_request.user_id else None,
                        session_id=str(state.conversation_id),
                        metadata={
                            "request_id": request_id,
                            "model": llm_request.model,
                            "tool_model": get_agent_settings().tool_model,
                            "router_prompt": get_query_router_prompt_version(),
                            "synthesis_prompt": SYNTHESIS_PROMPT_VERSION,
                        },
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
            _trace_io(input={"query": state.user_query_raw})

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

                _score("injection_score", float(injection_signal.score), "NUMERIC")
                _score("injection_severity", injection_signal.severity, "CATEGORICAL")

                if injection_signal.severity == "block":
                    GUARDRAIL_BLOCKS.labels("injection").inc()
                    refusal_text = (
                        "I'm sorry, but I can't process that request. "
                        "Please ask a financial question about your documents."
                    )
                    _trace_io(output={"answer": refusal_text, "route": "blocked"})
                    lf_mark(
                        _root_span,
                        "WARNING",
                        f"blocked by injection guardrail: {injection_signal.matched_rules}",
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

            # Prior questions were written to the tail raw at the API layer; the scan above
            # covered only the current one, so `prior_turns` scans each of them.
            state.prior_turns = prior_turns(
                state.context_messages,
                scan=get_injection_scan_user_input_enabled(),
                request_id=request_id,
            )

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

            # Scope isn't resolved yet, so the staleness check runs later, on the
            # synthesis path — the router only needs to know the data exists.
            prior_findings = _latest_findings_block(state.context_messages)
            prior_findings_present = prior_findings.block is not None
            router_input = RouterInput(
                query=state.user_query_raw,
                scope=chat_scope,
                prior_turns=state.prior_turns,
                prior_findings_block=prior_findings.block,
            )
            # Release before the router's LLM call. update_status above only flushes, and the
            # message reads reopen a transaction anyway, so without this the connection is
            # held across the router's LLM call. route_query commits again after its own
            # writes and reads, before each later LLM call. Measured as the residual
            # `idle in transaction` after the agent-loop fix: readiness audit §4.1.
            await session.commit()
            state.router_output, state.scope_result = await route_query(
                router_input,
                user_id=llm_request.user_id,
                llm_router=router,
                session=session,
                parent_request_id=llm_request.id,
                conversation_id=state.conversation_id,
            )
            ROUTER_DECISIONS.labels(state.router_output.route).inc()
            # Flushed by the commits below; expire_on_commit=False keeps the object usable.
            llm_request.query_shape = state.router_output.query_shape
            # The clarification card, when the client can show one: the question covers too
            # many companies, or names one that is ambiguous, unknown or outside the UI scope.
            card: dict | None = None
            if state.scope_result is not None:
                llm_request.scope_outcome = scope_outcome(state.router_output, state.scope_result)
                if allow_clarification and (
                    state.scope_result.too_broad_count is not None
                    or state.scope_result.clarifications
                ):
                    card = build_scope_clarification_event(
                        state.assistant_message_id,
                        state.scope_result,
                        named_companies=bool(state.router_output.entities),
                        max_companies=get_scope_max_companies(),
                    )
                    if state.scope_result.too_broad_count is None:
                        llm_request.scope_outcome = "clarification"
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
                    # The carried digest is the input a misroute has to be read against —
                    # without it a bad follow-up decision is undiagnosable from the trace.
                    input={
                        "query": state.user_query_raw,
                        "prior_findings_block": prior_findings.block,
                    },
                    output={
                        "route": state.router_output.route,
                        "query_shape": getattr(state.router_output, "query_shape", None),
                        "entities": [e.model_dump() for e in state.router_output.entities],
                        "user_intent": state.router_output.user_intent,
                        "scope": scope_summary,
                    },
                    metadata={
                        "scope": scope_summary,
                        "prior_findings_present": prior_findings_present,
                        "prior_findings_hops": prior_findings.hops,
                        "scope_source": state.scope_result.source if state.scope_result else None,
                        "unresolved_entities": state.scope_result.unresolved_entities
                        if state.scope_result
                        else [],
                        "scope_doc_ids": cap_list(_scope_doc_ids or []),
                        "scope_doc_count": len(_scope_doc_ids) if _scope_doc_ids is not None else 0,
                        "scope_per_entity_doc_ids": {
                            entity: {"count": len(ids), "doc_ids": cap_list(ids)}
                            for entity, ids in (_scope_per_entity or {}).items()
                        },
                        "scope_entity_manifest": cap_list(_scope_entity_manifest or []),
                    },
                )
                # Surface scope at the trace root so it's visible without drilling into
                # the route stage.
                if _root_span is not None:
                    with contextlib.suppress(Exception):
                        _root_span.update(metadata={"scope": scope_summary})  # type: ignore[attr-defined]

            # Early-exit: out_of_scope, the clarification card, or (with clarification off) a
            # question covering more companies than one run can analyse — skip RAG + LLM,
            # emit a fixed reply and persist
            too_broad = state.scope_result.too_broad_count if state.scope_result else None
            if (
                state.router_output.route == "out_of_scope"
                or card is not None
                or too_broad is not None
            ):
                if card is not None:
                    exit_reason = "clarification"
                    redirect_text = clarification_text(card)
                elif too_broad is not None:
                    exit_reason = "too_broad"
                    redirect_text = too_broad_response(too_broad, get_scope_max_companies())
                else:
                    exit_reason = "out_of_scope"
                    redirect_text = out_of_scope_response()
                _trace_io(output={"answer": redirect_text, "route": exit_reason})
                if card is not None:
                    await add_event(redis_app, request_id, "scope_clarification", card)
                await add_event(redis_app, request_id, "delta", {"text": redirect_text})
                await message_repo.update_on_final(
                    message_id=state.assistant_message_id,
                    content=redirect_text,
                    raw_content=redirect_text,
                    request_id=UUID(request_id),
                    # The card re-renders from here after a reload, and a reply re-runs
                    # this user message.
                    metadata_updates={
                        "kind": "clarification",
                        "clarification": card,
                        "user_message_id": str(llm_request.user_message_id),
                    }
                    if card is not None
                    else None,
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
                # A card stays out of the history: its re-run answers the same question, and
                # later turns should see question and answer as one pair.
                if card is None:
                    try:
                        await state.history.append_assistant(
                            state.conversation_id,
                            redirect_text,
                            state.assistant_seq,
                        )
                    except Exception:
                        logger.warning("chat_tail_append_failed", extra={"request_id": request_id})
                logger.info("pipeline.%s", exit_reason, extra={"request_id": request_id})
                return

            # Release the pgbouncer slot before the agent loop / carryover branch, which can
            # run for minutes — holding a transaction that long converts transaction pooling
            # into session pooling. `expire_on_commit=False` keeps llm_request/assistant_msg
            # usable after this.
            await session.commit()

            agent_settings = get_agent_settings()
            # `user_id` is nullable on LLMRequest, and the agent loop cannot search
            # without one — route that case to the no-context path explicitly.
            if state.router_output.route == "retrieval" and llm_request.user_id is not None:
                _tool_model_id: str = agent_settings.tool_model
                _tool_llm, *_tool_fallbacks = tool_model_chain(router, _tool_model_id)

                # Step 25: mark this request as agentic for DB queries/dashboards.
                # Left pending deliberately: flushing here would reopen the transaction the
                # commit above just closed, right before an LLM call. The dirty attribute
                # holds no connection, and the commit after the loop writes it.
                llm_request.request_type = "chat_agent"

                await _log_stage(
                    "agent_loop", model=_tool_llm.model_id, provider=_tool_llm.provider
                )

                _agent_lf_stack = contextlib.ExitStack()
                if lf:
                    _query_shape = getattr(state.router_output, "query_shape", None)
                    _agent_lf_stack.enter_context(
                        propagate_attributes(
                            metadata={
                                "agent_prompt": shape_config(_query_shape, agent_settings).prompt
                            }
                        )
                    )
                    # The `agent_loop` stage span opened by _log_stage is the agent's span;
                    # a second one nested inside it would only repeat it.
                    lf.update_current_span(
                        input={
                            "query": state.user_query_raw,
                            "query_shape": _query_shape,
                            "tool_model": _tool_model_id,
                        }
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
                        fallbacks=_tool_fallbacks,
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
                                # Is reporting incremental, or is the model one-shotting?
                                "plan_seeded": agent_meta.plan_seeded,
                                "plan_covered": agent_meta.plan_covered,
                                "report_calls_total": agent_meta.report_calls_total,
                                "turns_to_first_report": agent_meta.turns_to_first_report,
                                "unknown_aspect_keys": agent_meta.unknown_aspect_keys,
                                "unsearched_negatives": agent_meta.unsearched_negatives,
                                "uncited_claim_rate": agent_meta.uncited_claim_rate,
                                "search_arg_errors": agent_meta.search_arg_errors,
                                "report_parse_failures": agent_meta.report_parse_failures,
                            },
                            metadata={
                                "prompt_version": agent_meta.prompt_version,
                                "input_tokens_total": agent_meta.input_tokens_total,
                                "output_tokens_total": agent_meta.output_tokens_total,
                                "cost_usd_total": agent_meta.cost_usd_total,
                                "last_turn_input_tokens": agent_meta.last_turn_input_tokens,
                            },
                        )
                        if agent_meta.convergence_reason in _AGENT_FAILED_STOPS:
                            lf_mark_current(
                                "WARNING",
                                f"agent stopped early: {agent_meta.convergence_reason}"
                                f" (sealed={agent_meta.sealed})",
                            )
                        if agent_meta.plan_seeded:
                            _score(
                                "agent_plan_coverage",
                                agent_meta.plan_covered / agent_meta.plan_seeded,
                                "NUMERIC",
                            )
                        _score("agent_uncited_claim_rate", agent_meta.uncited_claim_rate, "NUMERIC")
                        _score(
                            "agent_convergence_reason",
                            agent_meta.convergence_reason or "none",
                            "CATEGORICAL",
                        )
                        _score("agent_sealed", float(agent_meta.sealed), "BOOLEAN")
                finally:
                    _agent_lf_stack.close()

                AGENT_ITERATIONS.observe(agent_meta.iterations)
                AGENT_STOP_REASONS.labels(
                    agent_meta.convergence_reason,
                    getattr(state.router_output, "query_shape", None) or "none",
                ).inc()
                state.agent_meta = agent_meta

                state.rag_context = agent_result.rag_context
                state.rag_context_str = agent_result.synthesis_context

                if agent_result.findings is not None:
                    agent_findings_json = agent_result.findings.model_dump_json()
                # Runaway guard only: real blocks are ~2-4k (rows are bounded by scope,
                # observations by max_plan_items). Clips at a line so a pathological
                # model output can't blow up the router prompt on every later turn.
                block = agent_result.findings_block
                if block is not None and len(block) > FINDINGS_BLOCK_MAX_CHARS:
                    block = block[:FINDINGS_BLOCK_MAX_CHARS].rsplit("\n", 1)[0] + "\n… (truncated)"
                state.findings_block = block
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
                # Answerable from what an earlier turn already retrieved — reformat,
                # restate, or arithmetic on a rate the user supplied. No excerpts here,
                # so the block is the only grounding the synthesis model gets.
                carried = _latest_findings_block(
                    state.context_messages, _scope_doc_ids, check_scope=True
                )
                FOLLOWUP_FINDINGS_CARRIED.labels(carried.outcome).inc()
                if carried.block:
                    state.rag_context_str = carried.block
                    state.findings_block = carried.block
                    state.findings_block_hops = carried.hops
                    state.findings_block_doc_ids = carried.doc_ids
                    state.answer_derived_from_carryover = True
                else:
                    state.rag_context_str = "(No document context - general question.)"
                FOLLOWUP_DIRECT_ANSWER.labels("true" if carried.block else "false").inc()
                logger.info(
                    "followup_findings_carryover",
                    extra={
                        "request_id": request_id,
                        "route": state.router_output.route,
                        "outcome": carried.outcome,
                        "hops": carried.hops,
                    },
                )

            # Release again before the synthesis stream.
            await session.commit()

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
                llm_chain = router.get_with_fallback(llm_request.model)
            except Exception as e:
                logger.exception("llm_router_error", extra={"request_id": request_id})
                lf_mark_current("ERROR", describe_error(e))
                lf_mark(_root_span, "ERROR", f"render_prompt: {describe_error(e)}")
                _trace_io(output={"error": describe_error(e)})
                await llm_request_repo.update_status(UUID(request_id), "failed")
                await add_event(redis_app, request_id, "error", error_event(e))
                await session.commit()
                return
            llm = llm_chain[0]

            prompt_version = SYNTHESIS_PROMPT_VERSION
            # Agent runs label by the router's shape; everything else answered without
            # retrieval, whatever shape the router guessed.
            ttft_shape = (
                (getattr(state.router_output, "query_shape", None) or "none")
                if state.agent_meta is not None
                else "direct"
            )
            ttft_s: float | None = None

            renderer = get_prompt_renderer()
            state.params = dict(llm_request.request_params or {})
            state.adapter_messages = assemble_prompt(
                prior=state.prior_turns,
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
                        "prior_turns": len(state.prior_turns),
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
            temperature = state.params.get("temperature")
            max_tokens = state.params.get("max_tokens")
            extra = {
                k: v for k, v in state.params.items() if k not in ("temperature", "max_tokens")
            }
            if lf:
                _gen = _gen_stack.enter_context(
                    lf.start_as_current_observation(
                        as_type="generation",
                        name="chat_model",
                        model=llm.model_id,
                        model_parameters=trace_params({**llm.default_params, **state.params}),
                        input=trace_input(state.adapter_messages, "chat_model"),
                    )
                )
            first_chunk_seen = False

            stream = FallbackStream(
                llm_chain,
                state.adapter_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **extra,
            )

            parser = BracketCitationParser()
            think_stripper = ThinkingStripper()

            try:
                chunk = None
                async for chunk in stream:
                    if not first_chunk_seen and chunk.text:
                        first_chunk_seen = True
                        # Langfuse derives time-to-first-token from this.
                        if _gen is not None:
                            with contextlib.suppress(Exception):
                                _gen.update(completion_start_time=datetime.now(UTC))  # type: ignore[attr-defined]
                    state.accumulated_content += chunk.text  # raw for DB
                    # Strip [S1] markers, track spans. Emitted for every chunk, including
                    # the final one — a final chunk can carry text, and skipping it would
                    # drop that text and any spans it completed from the SSE stream.
                    result = parser.feed(think_stripper.feed(chunk.text))
                    state.clean_content += result.visible_text
                    if result.visible_text:
                        if ttft_s is None:
                            ttft_s = _observe_ttft(llm_request.created_at, ttft_shape)
                        await add_event(
                            redis_app, request_id, "delta", {"text": result.visible_text}
                        )
                    for span in result.completed_spans:
                        await add_event(redis_app, request_id, "citation_span", span_to_dict(span))

                if stream.served is not llm:
                    # Fallback fired: record the model that actually answered, not the
                    # one originally requested, so cost/token metrics and the persisted
                    # request row aren't attributed to a model that never responded.
                    llm = stream.served
                    llm_request.model = llm.model_id
                    logger.warning(
                        "llm_fallback_served",
                        extra={"request_id": request_id, "served_model": llm.model_id},
                    )
                    if _gen is not None:
                        with contextlib.suppress(Exception):
                            _gen.update(model=llm.model_id)  # type: ignore[attr-defined]
                        lf_mark(_gen, "WARNING", f"fallback model {llm.model_id} answered")

                if chunk is not None:
                    final_result = parser.finalize()
                    state.clean_content += final_result.visible_text
                    if final_result.visible_text:
                        if ttft_s is None:
                            ttft_s = _observe_ttft(llm_request.created_at, ttft_shape)
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

                    # Carried to the next turn so a follow-up can be answered without
                    # re-retrieving. Unsealed findings are not carried — a partial result
                    # restated a turn later reads as settled fact.
                    if state.findings_block and (
                        state.agent_meta is None or state.agent_meta.sealed
                    ):
                        citation_meta["findings_block"] = state.findings_block
                        citation_meta["findings_block_hops"] = state.findings_block_hops
                        # A fresh run's block belongs to this turn's scope; a carried one
                        # keeps the scope it was originally retrieved under.
                        citation_meta["findings_block_doc_ids"] = (
                            state.findings_block_doc_ids
                            if state.answer_derived_from_carryover
                            else _scope_doc_ids
                        )
                    if state.answer_derived_from_carryover:
                        citation_meta["answer_derived_from_carryover"] = True

                    if agent_findings_json is not None:
                        citation_meta["agent_findings"] = agent_findings_json
                        # Persist the sealed/degraded marker beside the findings so a later
                        # turn's carry-over can tell a committed finalizer from a degraded
                        # partial serve (step 11 M2-synth) — `meta.sealed` is in-run only.
                        if state.agent_meta is not None:
                            citation_meta["agent_findings_sealed"] = state.agent_meta.sealed

                    # Finalize stage times (stream_llm_response ends here)
                    _elapsed = perf_counter() - stage_start
                    stage_times[current_stage] = round(_elapsed, 3)
                    CHAT_STAGE_DURATION.labels(current_stage).observe(_elapsed)
                    total_time = round(perf_counter() - pipeline_started_at, 3)

                    scores_are_rerank = (
                        state.agent_meta.scores_are_rerank if state.agent_meta else True
                    )
                    degraded = sorted(
                        state.agent_meta.degraded_capabilities if state.agent_meta else ()
                    )
                    confidence = compute_confidence(
                        top_score, num_chunks, scores_are_rerank=scores_are_rerank
                    )
                    # Citations are asked for only when excerpts were shown; a carried-over
                    # findings block is answered without them.
                    uncited_share = (
                        uncited_fact_share(state.clean_content, parser.all_spans)
                        if num_chunks
                        else None
                    )
                    ungrounded = bool(uncited_share)

                    # Build pipeline trace
                    trace_payload: dict = {
                        "v": 1,
                        "stage_times": stage_times,
                        "activity": await get_activity_log(redis_app, request_id),
                        "total_time": total_time,
                        "ttft_s": round(ttft_s, 3) if ttft_s is not None else None,
                        "config": {
                            "answer_model": llm.model_id,
                            "prompts": {
                                "router": get_query_router_prompt_version(),
                                "synthesis": prompt_version,
                            },
                        },
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
                            "prior_findings_present": prior_findings_present,
                            "prior_findings_carried": state.answer_derived_from_carryover,
                            "prior_findings_hops": state.findings_block_hops,
                        },
                    }
                    if state.agent_meta is not None:
                        m = state.agent_meta
                        trace_payload["config"]["tool_model"] = agent_settings.tool_model
                        trace_payload["config"]["prompts"]["agent"] = m.prompt_version
                        trace_payload["agent"] = {
                            "iterations": m.iterations,
                            "tool_calls_total": m.tool_calls_total,
                            "convergence_reason": m.convergence_reason,
                            "sealed": m.sealed,
                            "currency_normalized": state.agent_currency_converted,
                            "answer_entity": state.agent_answer_entity,
                            "fx_rates_used": state.agent_fx_rates,
                            # Decomposition/coverage instrumentation, persisted so DB and
                            # Grafana queries over Message.trace can see it.
                            "plan_seeded": m.plan_seeded,
                            "plan_covered": m.plan_covered,
                            "report_calls_total": m.report_calls_total,
                            "turns_to_first_report": m.turns_to_first_report,
                            "unknown_aspect_keys": m.unknown_aspect_keys,
                            "unsearched_negatives": m.unsearched_negatives,
                            "uncited_claim_rate": m.uncited_claim_rate,
                            "search_arg_errors": m.search_arg_errors,
                            "report_parse_failures": m.report_parse_failures,
                            "last_turn_input_tokens": m.last_turn_input_tokens,
                        }
                    trace_payload["guardrails"] = {
                        "confidence": confidence,
                        "top_reranker_score": top_score,
                        "scores_are_rerank": scores_are_rerank,
                        "num_chunks": num_chunks,
                        "degraded_retrieval": degraded,
                        "ungrounded_claims": ungrounded,
                        "uncited_fact_share": uncited_share,
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

                    # Persist the answer-quality signals onto the message itself, not just
                    # the trace: the SSE `metadata` event only reaches the client that was
                    # streaming, so without this every badge vanishes on reload.
                    citation_meta["confidence"] = confidence
                    citation_meta["ungrounded_claims"] = ungrounded
                    citation_meta["route"] = (
                        state.router_output.route if state.router_output else None
                    )
                    turn_summary = _turn_summary(state.router_output, state.scope_result)
                    if turn_summary is not None:
                        citation_meta["turn_summary"] = turn_summary.model_dump()
                    if degraded:
                        citation_meta["degraded_retrieval"] = degraded

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
                        observe_llm_latency(_model, llm_request.request_type, chunk.stats)
                        await llm_request_repo.update_on_final(
                            request_id=UUID(request_id),
                            **stats_to_request_kwargs(chunk.stats),
                            trace_id=lf_trace_id,
                        )
                    if _gen is not None:
                        with contextlib.suppress(Exception):
                            _gen.update(  # type: ignore[union-attr]
                                output=state.accumulated_content, **trace_usage(chunk.stats)
                            )
                    _trace_io(
                        output={
                            "answer": state.clean_content,
                            "route": state.router_output.route,
                            "confidence": confidence,
                            "ungrounded_claims": ungrounded,
                            "references": len(ref_items),
                        }
                    )
                    _score("confidence", confidence, "CATEGORICAL")
                    _score("ungrounded_claims", float(ungrounded), "BOOLEAN")
                    if uncited_share is not None:
                        _score("uncited_fact_share", uncited_share, "NUMERIC")
                    _score("retrieval_degraded", float(bool(degraded)), "BOOLEAN")
                    if degraded:
                        lf_mark(_root_span, "WARNING", f"degraded retrieval: {degraded}")
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
                            "degraded_retrieval": degraded,
                        },
                    )
                    logger.info(
                        "confidence_score",
                        extra={
                            "request_id": request_id,
                            "confidence": confidence,
                            "top_score": top_score,
                            "scores_are_rerank": scores_are_rerank,
                            "num_chunks": num_chunks,
                            "degraded_retrieval": degraded,
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
                            findings_block=citation_meta.get("findings_block"),
                            answer_derived_from_carryover=state.answer_derived_from_carryover,
                            findings_block_hops=state.findings_block_hops,
                            findings_block_doc_ids=citation_meta.get("findings_block_doc_ids"),
                            turn_summary=turn_summary,
                        )
                    except Exception:
                        logger.warning("chat_tail_append_failed", extra={"request_id": request_id})

                    # Auto-name conversation on its first real answer: no earlier assistant
                    # message in history, which already leaves out clarification cards. A card
                    # can push that answer past seq 2, after a reply or a re-sent question.
                    # Must emit conversation_title BEFORE the usage event, since the frontend
                    # stops reading the stream as soon as it receives usage (the final sentinel).
                    naming_cfg = get_conversation_naming_config()
                    if (
                        naming_cfg["enabled"]
                        and state.user_query_raw
                        and not any(
                            m.role == schemas.Role.assistant for m in state.context_messages
                        )
                    ):
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
                lf_mark(_gen, "ERROR", describe_error(e))
                lf_mark_current("ERROR", describe_error(e))
                lf_mark(_root_span, "ERROR", f"{current_stage}: {describe_error(e)}")
                _trace_io(output={"error": describe_error(e)})
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
        PIPELINE_ERRORS.labels(current_stage).inc()
        logger.exception(
            "pipeline.failed_at_stage",
            extra={"request_id": request_id, "stage": current_stage},
        )
        # The stage span is still the current one; it closes in the finally below.
        lf_mark_current("ERROR", describe_error(exc))
        lf_mark(_root_span, "ERROR", f"{current_stage}: {describe_error(exc)}")
        _trace_io(output={"error": describe_error(exc)})
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
        if current_stage != "initializing" and current_stage_event_id is not None:
            _, end_data = build_activity_event("stage_ended", event_id=current_stage_event_id)
            await add_event(redis_app, request_id, "activity", end_data)
        # After the last add_event above — an XADD on an expired/expiring key recreates it
        # without a TTL, so this must be the final write to the stream.
        with contextlib.suppress(Exception):
            await expire_event_stream(redis_app, request_id)
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
