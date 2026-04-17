"""Chat pipeline task."""

from __future__ import annotations

import asyncio
import logging
from time import perf_counter
from uuid import UUID

from celery.signals import setup_logging, worker_process_init, worker_process_shutdown
from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from src.api.logging import configure_worker_logging
from src.celery_app import celery_app
from src.models.message import Message, MessageStatus
from src.redis_client import add_event
from src.repository import (
    ConversationRepository,
    DocumentRepository,
    LLMRequestRepository,
    MessageRepository,
)
from src.repository.llm_request_repository import stats_to_request_kwargs
from src.schemas import chat as schemas
from src.schemas.chat import ChatPipelineState
from src.schemas.query_router import ChatScope, RouterInput
from src.schemas.query_transform import ScopeDocSummary, TransformedQuery, TransformerInput
from src.schemas.retrieval import ProcessedQuery, RetrievalTrace
from src.services.chat.citation_parser import BracketCitationParser
from src.services.chat.events import (
    ThinkingStripper,
    build_all_references,
    build_references_list,
    build_usage_event,
    error_event,
    out_of_scope_response,
    span_to_dict,
)
from src.services.context import ConversationHistory, assemble_prompt
from src.services.llm_router import LLMRouter, get_router
from src.services.prompts.prompt_renderer import get_prompt_renderer, get_system_prompt
from src.services.retrieval.chat_rag import run_chat_rag_pipeline

# from src.services.retrieval.query_processor import process_query
from src.services.retrieval.query_transformer import transform_query
from src.services.retrieval.reranker import Reranker, get_reranker
from src.services.router.router import route_query
from src.utils.config import get_db_url, get_query_transformer_config, get_redis_app_url

logger = logging.getLogger(__name__)

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
    _router = None
    _reranker = None
    _redis_app = None
    _engine = None
    _session_factory = None
    _worker_loop.close()
    _worker_loop = None


async def _run_chat_pipeline(request_id: str) -> None:
    if _session_factory is None or _redis_app is None:
        raise RuntimeError("Chat worker resources are not initialized")

    sf = _session_factory
    redis_app = _redis_app
    pipeline_started_at = perf_counter()
    stage_start = perf_counter()
    stage_times: dict[str, float] = {}
    retrieval_trace: RetrievalTrace | None = None
    stage_total = 8
    stage_index = 0
    current_stage = "initializing"

    def _log_stage(stage_name: str) -> None:
        nonlocal stage_index, current_stage, stage_start
        if current_stage != "initializing":
            stage_times[current_stage] = round(perf_counter() - stage_start, 3)
        stage_index += 1
        current_stage = f"{stage_index:02d}_{stage_name}"
        stage_start = perf_counter()
        logger.info(
            f"pipeline.stage [{stage_index}/{stage_total}] {stage_name}",
            extra={"request_id": request_id, "stage": stage_name},
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
            _log_stage("load_and_validate_request")
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

            # 2. build_conversation_context
            _log_stage("build_conversation_context")
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

            # 3. route_query
            _log_stage("route_query")
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
            # Shim for downstream stages that still read processed_query.route
            state.processed_query = ProcessedQuery(
                normalized_text=state.user_query_raw.strip(),
                route="retrieve"
                if state.router_output.route == "retrieval"
                else state.router_output.route,
                user_intent=state.router_output.user_intent,
                reason=state.router_output.reasoning,
            )
            logger.info(
                "rag_route",
                extra={
                    "request_id": request_id,
                    "route": state.router_output.route,
                    "entities": [e.model_dump() for e in state.router_output.entities],
                    "user_intent": state.router_output.user_intent,
                    "needs_decomposition": state.router_output.needs_decomposition,
                    "scope_source": state.scope_result.source if state.scope_result else None,
                },
            )

            # Early-exit: out_of_scope — skip RAG + LLM, emit redirect and persist
            if state.processed_query.route == "out_of_scope":
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
                    redirect_text,
                    None,
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

            # 3.5 transform_query — retrieval route only
            if state.processed_query.route == "retrieve" and llm_request.user_id:
                _log_stage("transform_query")
                doc_ids_scope = state.scope_result.doc_ids if state.scope_result else None
                per_entity = state.scope_result.per_entity_doc_ids if state.scope_result else None
                known_entity_names: list[str] = list(per_entity.keys()) if per_entity else []

                # Decomposition guard: disable when fewer than 2 resolved entities
                needs_decomp = state.router_output.needs_decomposition
                decomp_overridden = False
                if needs_decomp and (not per_entity or len(per_entity) < 2):
                    needs_decomp = False
                    decomp_overridden = True
                    logger.info(
                        "decomposition_overridden",
                        extra={
                            "request_id": request_id,
                            "reason": "insufficient_entities",
                            "entity_count": len(per_entity) if per_entity else 0,
                        },
                    )

                scope_docs: list[ScopeDocSummary] = []
                if doc_ids_scope:
                    cfg = get_query_transformer_config()
                    doc_repo = DocumentRepository(session)
                    rows = await doc_repo.get_scope_doc_summaries(
                        llm_request.user_id, doc_ids_scope, limit=cfg["max_scope_docs"]
                    )
                    scope_docs = [
                        ScopeDocSummary(document_id=r[0], company=r[1], year=r[2]) for r in rows
                    ]

                transformer_input = TransformerInput(
                    user_query_raw=state.user_query_raw,
                    conversation_history=[
                        {"role": m.role.value, "content": m.content}
                        for m in (state.context_messages or [])
                    ],
                    router_entities=state.router_output.entities,
                    user_intent=state.router_output.user_intent,
                    needs_decomposition=needs_decomp,
                    scope_docs=scope_docs,
                    known_entity_names=known_entity_names,
                )
                try:
                    state.transformed_query = await transform_query(
                        transformer_input,
                        llm_router=router,
                        session=session,
                        parent_request_id=llm_request.id,
                        conversation_id=state.conversation_id,
                        user_id=llm_request.user_id,
                    )
                    if decomp_overridden:
                        state.transformed_query.decomposition_overridden = True
                except Exception:
                    logger.exception("query_transform_failed", extra={"request_id": request_id})
                    state.transformed_query = TransformedQuery(
                        semantic_query=state.user_query_raw,
                        keyword_query=state.user_query_raw,
                        fallback=True,
                    )

                tq = state.transformed_query
                logger.info(
                    "query_transform_result",
                    extra={
                        "request_id": request_id,
                        "semantic_query": tq.semantic_query,
                        "keyword_query": tq.keyword_query,
                        "fallback": tq.fallback,
                        "decomposition_overridden": tq.decomposition_overridden,
                        "sub_query_count": len(tq.sub_queries),
                        "sub_queries": [
                            {
                                "focus_entity": s.focus_entity,
                                "semantic_query": s.semantic_query,
                                "keyword_query": s.keyword_query,
                                "entity_match_quality": s.entity_match_quality,
                            }
                            for s in tq.sub_queries
                        ],
                    },
                )

            # 4. build_rag_context
            _log_stage("build_rag_context")
            if state.processed_query.route == "direct_answer" or not llm_request.user_id:
                state.rag_context_str = "(No document context - general question.)"
            else:
                doc_ids = state.scope_result.doc_ids if state.scope_result is not None else None
                transformed = state.transformed_query or TransformedQuery(
                    semantic_query=state.user_query_raw,
                    keyword_query=state.user_query_raw,
                    fallback=True,
                )
                per_entity = state.scope_result.per_entity_doc_ids if state.scope_result else None
                sub_doc_ids: list[list[UUID] | None] | None = None
                if transformed.sub_queries and per_entity:
                    sub_doc_ids = []
                    for sub in transformed.sub_queries:
                        ids = per_entity.get(sub.focus_entity)
                        if not ids:
                            logger.warning(
                                "subquery_entity_unresolved",
                                extra={
                                    "request_id": request_id,
                                    "focus_entity": sub.focus_entity,
                                    "match_quality": sub.entity_match_quality,
                                },
                            )
                        if ids:
                            sub_doc_ids.append(ids)

                state.rag_context, retrieval_trace = await run_chat_rag_pipeline(
                    session,
                    transformed=transformed,
                    user_id=llm_request.user_id,
                    doc_ids=doc_ids,
                    sub_doc_ids=sub_doc_ids,
                    reranker=_get_reranker(),
                )
                state.rag_context_str = (
                    state.rag_context.formatted_context
                    or "(No document context - general question.)"
                )

            # 5. render_prompt
            _log_stage("render_prompt")
            # Resolve model + citation_mode before rendering so prompt version is model-aware
            try:
                llm = router.get(llm_request.model)
            except Exception as e:
                logger.exception("llm_router_error", extra={"request_id": request_id})
                await llm_request_repo.update_status(UUID(request_id), "failed")
                await add_event(redis_app, request_id, "error", error_event(e))
                await session.commit()
                return

            citation_mode = llm.capabilities.get("citation_mode", "none")
            prompt_version = "v3_bracket" if citation_mode == "bracket" else "v3_none"

            renderer = get_prompt_renderer()
            state.params = dict(llm_request.request_params or {})
            state.adapter_messages = assemble_prompt(
                history=state.context_messages,
                system_prompt=get_system_prompt(version=prompt_version),
                rag_context=state.rag_context_str,
                user_query=state.user_query_raw,
                renderer=renderer,
            )

            # 6. stream_llm_response
            _log_stage("stream_llm_response")
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

            parser = BracketCitationParser() if citation_mode == "bracket" else None
            think_stripper = ThinkingStripper()

            try:
                async for chunk in stream:
                    state.accumulated_content += chunk.text  # raw for DB
                    visible_chunk = think_stripper.feed(chunk.text)

                    if parser is not None:
                        # Bracket-citation mode: strip [S1] markers, track spans
                        result = parser.feed(visible_chunk)
                        state.clean_content += result.visible_text

                        if not chunk.is_final:
                            if result.visible_text:
                                await add_event(
                                    redis_app, request_id, "delta", {"text": result.visible_text}
                                )
                            for span in result.completed_spans:
                                labels = parser.label_map.get_labels_for_refs(span.ref_ids)
                                await add_event(
                                    redis_app,
                                    request_id,
                                    "citation_span",
                                    span_to_dict(span, labels),
                                )
                            continue

                        # ── Final chunk (bracket mode) ──
                        final_result = parser.finalize()
                        state.clean_content += final_result.visible_text
                        if final_result.visible_text:
                            await add_event(
                                redis_app, request_id, "delta", {"text": final_result.visible_text}
                            )
                        for span in final_result.completed_spans:
                            labels = parser.label_map.get_labels_for_refs(span.ref_ids)
                            await add_event(
                                redis_app,
                                request_id,
                                "citation_span",
                                span_to_dict(span, labels),
                            )

                        # Emit references: only cited sources
                        if state.rag_context and parser.label_map.mapping:
                            ref_items = build_references_list(state.rag_context, parser.label_map)
                            await add_event(
                                redis_app, request_id, "references", {"items": ref_items}
                            )

                    else:
                        # No-citation mode: raw text = clean text, no span parsing
                        state.clean_content += visible_chunk

                        if not chunk.is_final:
                            if visible_chunk:
                                await add_event(
                                    redis_app, request_id, "delta", {"text": visible_chunk}
                                )
                            continue

                        # ── Final chunk (no-citation mode) ──
                        # Emit ALL retrieved sources as evidence panel references
                        if state.rag_context and state.rag_context.items:
                            ref_items = build_all_references(state.rag_context)
                            await add_event(
                                redis_app, request_id, "references", {"items": ref_items}
                            )

                    # 7. persist_and_emit
                    _log_stage("persist_and_emit")
                    # Build citation metadata for persistence
                    citation_meta: dict = {}
                    if parser is not None:
                        if parser.all_spans:
                            citation_meta["citation_spans"] = [
                                span_to_dict(s, parser.label_map.get_labels_for_refs(s.ref_ids))
                                for s in parser.all_spans
                            ]
                        if state.rag_context and parser.label_map.mapping:
                            citation_meta["references"] = build_references_list(
                                state.rag_context, parser.label_map
                            )
                    elif state.rag_context and state.rag_context.items:
                        citation_meta["references"] = build_all_references(state.rag_context)

                    if state.rag_context and state.rag_context.items:
                        citation_meta["retrieved_chunks"] = [
                            {"chunk_id": str(item.chunk_id), "score": item.score}
                            for item in state.rag_context.items
                        ]

                    # Finalize stage times (stream_llm_response ends here)
                    stage_times[current_stage] = round(perf_counter() - stage_start, 3)
                    total_time = round(perf_counter() - pipeline_started_at, 3)

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
                            "needs_decomposition": state.router_output.needs_decomposition,
                            "entities": [e.model_dump() for e in state.router_output.entities],
                            "scope_source": state.scope_result.source
                            if state.scope_result
                            else None,
                        },
                    }
                    if state.transformed_query is not None:
                        tq = state.transformed_query
                        trace_payload["query_transform"] = {
                            "semantic_query": tq.semantic_query,
                            "keyword_query": tq.keyword_query,
                            "sub_queries": [sq.model_dump() for sq in tq.sub_queries],
                            "fallback": tq.fallback,
                            "decomposition_overridden": tq.decomposition_overridden,
                        }
                    if retrieval_trace is not None:
                        trace_payload["retrieval"] = retrieval_trace.model_dump(exclude_none=True)

                    await message_repo.update_on_final(
                        message_id=state.assistant_message_id,
                        content=state.clean_content,
                        raw_content=state.accumulated_content,
                        request_id=UUID(request_id),
                        metadata_updates=citation_meta or None,
                        trace=trace_payload,
                    )
                    if chunk.stats:
                        await llm_request_repo.update_on_final(
                            request_id=UUID(request_id),
                            **stats_to_request_kwargs(chunk.stats),
                        )
                    await llm_request_repo.update_status(UUID(request_id), "completed")
                    await conversation_repo.update_on_message(
                        conversation_id=state.conversation_id,
                        message_id=state.assistant_message_id,
                        new_seq=state.assistant_seq,
                    )

                    usage_data = build_usage_event(
                        state.accumulated_content,
                        state.rag_context,
                        state.assistant_message_id,
                        state.assistant_seq,
                        chunk.stats,
                        citation_spans=parser.all_spans if parser is not None else None,
                        label_map=parser.label_map if parser is not None else None,
                    )
                    await add_event(redis_app, request_id, "usage", usage_data)
                    await session.commit()

                    try:
                        await state.history.append_assistant(
                            state.conversation_id,
                            state.clean_content,
                            state.assistant_seq,
                        )
                    except Exception:
                        logger.warning("chat_tail_append_failed", extra={"request_id": request_id})

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
                await add_event(redis_app, request_id, "error", error_event(e, str(e)))
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


@celery_app.task(bind=True, name="process_chat", acks_late=True, reject_on_worker_lost=True)
def process_chat(_self, request_id: str) -> None:
    """Celery task: process chat request."""
    _initialize_worker_resources()
    loop = _get_worker_loop()
    loop.run_until_complete(_run_chat_pipeline(request_id))
