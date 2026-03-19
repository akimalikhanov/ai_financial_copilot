"""Chat pipeline task."""

from __future__ import annotations

import asyncio
import logging
from decimal import Decimal
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
from src.redis_client import add_event, append_chat_tail, cas_populate_chat_tail, get_chat_tail
from src.repository import (
    ConversationRepository,
    LLMRequestRepository,
    MessageRepository,
)
from src.schemas import chat as schemas
from src.schemas.chat import ChatPipelineState
from src.services.chat.events import build_usage_event, error_event
from src.services.context import build_context
from src.services.llm_adapters.base_adapter import ChatMessage as AdapterChatMessage
from src.services.llm_adapters.base_adapter import Role as AdapterRole
from src.services.llm_router import LLMRouter, get_router
from src.services.prompts.prompt_renderer import get_prompt_renderer, get_system_prompt
from src.services.retrieval.chat_rag import resolve_doc_ids, run_chat_rag_pipeline
from src.services.retrieval.query_processor import process_query
from src.utils.config import get_chat_tail_max_messages, get_db_url, get_redis_app_url

logger = logging.getLogger(__name__)

_worker_loop: asyncio.AbstractEventLoop | None = None
_redis_app: Redis | None = None
_engine = None
_session_factory: async_sessionmaker[AsyncSession] | None = None
_router: LLMRouter | None = None


def _initialize_worker_resources() -> None:
    global _worker_loop, _redis_app, _engine, _session_factory, _router
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


def _get_worker_loop() -> asyncio.AbstractEventLoop:
    if _worker_loop is None or _worker_loop.is_closed():
        raise RuntimeError("Chat worker loop is not initialized")
    return _worker_loop


def _get_router() -> LLMRouter:
    if _router is None:
        raise RuntimeError("Chat worker LLM router is not initialized")
    return _router


@setup_logging.connect
def _on_celery_setup_logging(**_kwargs: object) -> None:
    configure_worker_logging()


@worker_process_init.connect
def _on_worker_process_init(**_kwargs: object) -> None:
    global _worker_loop, _redis_app, _engine, _session_factory, _router
    configure_worker_logging()
    _initialize_worker_resources()


@worker_process_shutdown.connect
def _on_worker_process_shutdown(**_kwargs: object) -> None:
    global _worker_loop, _redis_app, _engine, _session_factory, _router
    if _worker_loop is None or _worker_loop.is_closed():
        return
    if _router is not None:
        _worker_loop.run_until_complete(_router.close())
    if _redis_app is not None:
        _worker_loop.run_until_complete(_redis_app.aclose())
    if _engine is not None:
        _worker_loop.run_until_complete(_engine.dispose())
    _router = None
    _redis_app = None
    _engine = None
    _session_factory = None
    _worker_loop.close()
    _worker_loop = None


def _to_adapter_messages(messages: list[schemas.ChatMessage]) -> list[AdapterChatMessage]:
    return [
        AdapterChatMessage(
            role=AdapterRole(m.role),
            content=m.content,
            name=m.name,
            tool_call_id=m.tool_call_id,
        )
        for m in messages
    ]


async def _run_chat_pipeline(request_id: str) -> None:
    if _session_factory is None or _redis_app is None:
        raise RuntimeError("Chat worker resources are not initialized")

    sf = _session_factory
    redis_app = _redis_app
    pipeline_started_at = perf_counter()
    stage_start = perf_counter()
    stage_times: dict[str, float] = {}
    stage_total = 7
    stage_index = 0
    current_stage = "initializing"

    def _log_stage(stage_name: str) -> None:
        nonlocal stage_index, current_stage, stage_start
        if current_stage != "initializing":
            stage_times[current_stage] = round(perf_counter() - stage_start, 3)
        stage_index += 1
        current_stage = stage_name
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
            user_seq = llm_request.snapshot_seq or 0
            cached = await get_chat_tail(redis_app, str(state.conversation_id))
            if cached is not None:
                cached_msgs, cached_seq = cached
                if cached_seq >= user_seq:
                    try:
                        state.context_messages = [
                            schemas.ChatMessage.model_validate(m) for m in cached_msgs
                        ]
                    except Exception:
                        logger.warning(
                            "invalid_chat_tail_cache",
                            extra={"conversation_id": str(state.conversation_id)},
                        )

            if state.context_messages is None:
                state.context_messages, latest_seq = await build_context(
                    message_repo,
                    state.conversation_id,
                    before_seq=state.assistant_seq,
                    max_messages=get_chat_tail_max_messages(),
                )
                await cas_populate_chat_tail(
                    redis_app,
                    str(state.conversation_id),
                    [m.model_dump(mode="json") for m in state.context_messages],
                    latest_seq,
                )

            last_user = next(
                (m for m in reversed(state.context_messages) if m.role == schemas.Role.user),
                None,
            )
            state.user_query_raw = last_user.content if last_user else ""

            # 3. route_query
            _log_stage("route_query")
            router = _get_router()
            state.processed_query = await process_query(state.user_query_raw, router=router)
            logger.info(
                "rag_route", extra={"request_id": request_id, "route": state.processed_query.route}
            )

            # 4. build_rag_context
            _log_stage("build_rag_context")
            if state.processed_query.route == "direct_answer" or not llm_request.user_id:
                state.rag_context_str = "(No document context - general question.)"
            else:
                doc_ids = resolve_doc_ids(llm_request.user_id)
                state.rag_context = await run_chat_rag_pipeline(
                    session,
                    state.processed_query.normalized_text,
                    llm_request.user_id,
                    doc_ids,
                )
                state.rag_context_str = (
                    state.rag_context.formatted_context
                    or "(No document context - general question.)"
                )

            # 5. render_prompt
            _log_stage("render_prompt")
            renderer = get_prompt_renderer()
            rendered_user = renderer.render_user_message(
                context=state.rag_context_str,
                user_query=state.user_query_raw,
            )
            modified = list(state.context_messages)
            if modified and modified[-1].role == schemas.Role.user:
                modified[-1] = schemas.ChatMessage(role=schemas.Role.user, content=rendered_user)
            else:
                modified.append(schemas.ChatMessage(role=schemas.Role.user, content=rendered_user))

            state.params = dict(llm_request.request_params or {})
            state.adapter_messages = [
                AdapterChatMessage(role=AdapterRole.system, content=get_system_prompt()),
            ] + _to_adapter_messages(modified)

            # 6. stream_llm_response
            _log_stage("stream_llm_response")
            try:
                llm = router.get(llm_request.model)
            except Exception as e:
                logger.exception("llm_router_error", extra={"request_id": request_id})
                await llm_request_repo.update_status(UUID(request_id), "failed")
                await add_event(redis_app, request_id, "error", error_event(e))
                await session.commit()
                return

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

            try:
                async for chunk in stream:
                    state.accumulated_content += chunk.text
                    if not chunk.is_final:
                        await add_event(redis_app, request_id, "delta", {"text": chunk.text})
                        continue

                    # 7. persist_and_emit
                    _log_stage("persist_and_emit")
                    await message_repo.update_on_final(
                        message_id=state.assistant_message_id,
                        content=state.accumulated_content,
                        request_id=UUID(request_id),
                    )
                    if chunk.stats:
                        await llm_request_repo.update_on_final(
                            request_id=UUID(request_id),
                            prompt_tokens=chunk.stats.input_tokens,
                            completion_tokens=chunk.stats.output_tokens,
                            reasoning_tokens=chunk.stats.reasoning_tokens,
                            total_tokens=chunk.stats.total_tokens,
                            cost_usd=(
                                Decimal(str(chunk.stats.cost_usd))
                                if chunk.stats.cost_usd is not None
                                else None
                            ),
                            latency_ms=(
                                int(chunk.stats.latency_ms)
                                if chunk.stats.latency_ms is not None
                                else None
                            ),
                            ttft_ms=(
                                int(chunk.stats.ttft_ms)
                                if chunk.stats.ttft_ms is not None
                                else None
                            ),
                            tps=int(chunk.stats.tps) if chunk.stats.tps is not None else None,
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
                    )
                    await add_event(redis_app, request_id, "usage", usage_data)
                    await session.commit()

                    try:
                        await append_chat_tail(
                            redis_app,
                            str(state.conversation_id),
                            schemas.ChatMessage(
                                role=schemas.Role.assistant,
                                content=state.accumulated_content,
                            ).model_dump(mode="json"),
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
