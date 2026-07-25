"""The agent tool-calling loop.

Turn control flow is explicit: `_run_turn` returns a `TurnOutcome` (`Continue`,
`Finalize`, `Stop`) instead of the old ~8 scattered `convergence_reason = ...; break`
sites threaded through one function. This is a real control-flow rewrite, not a
mechanical move — see the doc's honesty note on this step.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from dataclasses import replace as dc_replace
from time import perf_counter
from typing import TYPE_CHECKING
from uuid import UUID

from pydantic import ValidationError
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.observability import langfuse as lf_client
from src.observability.metrics import (
    AGENT_TOOL_CALLS,
    AGENT_TOOL_DURATION,
    LLM_CACHE_HIT_TOKENS,
    LLM_COST,
    LLM_TOKENS,
)
from src.redis_client import add_event
from src.repository.llm_request_repository import LLMRequestRepository, stats_to_request_kwargs
from src.schemas.agent_findings import AgentFindings, AnalyticalFindings
from src.schemas.query_transform import ScopeDocSummary, TransformedQuery
from src.schemas.retrieval import ChunkPromptPayload, RetrievedChunk
from src.services.chat.agent import gates as gates_module
from src.services.chat.agent import tools as tools_module
from src.services.chat.agent.state import (
    AgentLoopMeta,
    AgentRunState,
    ConvergenceReason,
    EffortPrior,
    build_meta,
    get_agent_settings,
)
from src.services.chat.agent.tools import SearchDocumentsArgs
from src.services.chat.agent.transcript import Transcript
from src.services.llm_adapters.base_adapter import (
    AssistantTurnResult,
    ChatMessage,
    LLMResponseStats,
    Role,
    ToolCallRef,
)
from src.services.prompts.prompt_renderer import get_system_prompt
from src.services.retrieval.chat_rag import run_chat_rag_pipeline
from src.services.retrieval.payload_hydrator import get_chunk_prompt_payloads
from src.services.retrieval.query_transformer import rewrite_query
from src.utils.config import get_query_transformer_model

if TYPE_CHECKING:
    from src.schemas.chat import ChatPipelineState
    from src.services.llm_router import RoutedLLM
    from src.services.retrieval.reranker import Reranker

logger = logging.getLogger(__name__)


@dataclass
class _SearchResult:
    entity: str
    chunks: list[RetrievedChunk]
    # Hydrated payloads for chunks — context is assembled later, sequentially, so
    # S-labels can be numbered globally across all searches in the request.
    payloads: dict[UUID, ChunkPromptPayload]
    error_str: str | None = None
    rewrite_stats: LLMResponseStats | None = None


ExecuteSearchFn = Callable[
    [ToolCallRef, "ChatPipelineState", AsyncSession, "Reranker | None", Redis, str, int],
    Awaitable[_SearchResult],
]


# ---------------------------------------------------------------------------
# Turn outcome
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Continue:
    pass


@dataclass(frozen=True)
class Finalize:
    findings: AgentFindings | AnalyticalFindings


@dataclass(frozen=True)
class Stop:
    reason: ConvergenceReason


TurnOutcome = Continue | Finalize | Stop


# ---------------------------------------------------------------------------
# Search execution
# ---------------------------------------------------------------------------


async def _execute_search(
    tc: ToolCallRef,
    state: ChatPipelineState,
    session: AsyncSession,
    reranker: Reranker | None,
    redis_app: Redis,
    request_id: str,
    iteration: int,
) -> _SearchResult:
    try:
        search_args = SearchDocumentsArgs.model_validate_json(tc.arguments)
    except ValidationError:
        logger.warning(
            "agent_search_args_invalid",
            extra={"request_id": request_id, "raw_args": tc.arguments[:500]},
        )
        AGENT_TOOL_CALLS.labels("search_documents", "error").inc()
        return _SearchResult(
            entity="",
            chunks=[],
            payloads={},
            error_str="search_documents call had invalid arguments — entity and query are required strings.",
        )
    entity = search_args.entity
    raw_query = search_args.query

    _tool_started = perf_counter()
    await add_event(redis_app, request_id, "tool_call_started", {"entity": entity})

    # Resolve doc_ids for this entity
    per_entity = (state.scope_result.per_entity_doc_ids or {}) if state.scope_result else {}
    if entity and entity in per_entity:
        doc_ids = per_entity[entity]
    elif not entity and per_entity:
        # Analytical agent passes entity="" — scope to the first (primary) entity's docs
        # rather than leaking to the full user corpus.
        doc_ids = next(iter(per_entity.values()))
    else:
        doc_ids = state.scope_result.doc_ids if state.scope_result else None

    # Rewrite at tool boundary — cheap model, eval-independent
    scope_docs: list[ScopeDocSummary] = []
    if state.scope_result and state.scope_result.entity_manifest:
        for item in state.scope_result.entity_manifest:
            if item.entity_name == entity:
                scope_docs = [
                    ScopeDocSummary(
                        document_id=s["doc_id"],
                        company=entity,
                        year=s.get("year"),
                    )
                    for s in (item.doc_summaries or [])
                ]
                break

    lf = lf_client.get_client()
    _search_lf_stack = contextlib.ExitStack()
    if lf:
        _search_lf_stack.enter_context(
            lf.start_as_current_observation(
                as_type="retriever",
                name=f"tool_search_{entity}_{iteration}",
                input={
                    "entity": entity,
                    "query": raw_query,
                    # Show the resolved scope this search was constrained to, so the
                    # trace makes clear which docs the agent could actually see.
                    "scope_doc_ids": [str(d) for d in doc_ids] if doc_ids else "all",
                    "scope_doc_count": len(doc_ids) if doc_ids else "all",
                    "scoped_via_entity": entity in per_entity,
                },
            )
        )

    rewrite_stats: LLMResponseStats | None = None
    try:
        transformed, rewrite_stats = await rewrite_query(
            raw_query,
            scope_docs=scope_docs or None,
            session=session,
            parent_request_id=state.llm_request.id if state.llm_request else None,
            conversation_id=state.conversation_id,
            user_id=state.llm_request.user_id if state.llm_request else None,
            extra_request_params={"entity": entity, "iteration": iteration, "source": "agent"},
        )
    except Exception:
        logger.warning("agent_rewrite_failed", extra={"entity": entity, "query": raw_query})
        transformed = TransformedQuery(
            semantic_query=raw_query,
            keyword_query=raw_query,
            fallback=True,
        )
    try:
        _, _, raw_chunks = await run_chat_rag_pipeline(
            session,
            transformed=transformed,
            user_id=state.llm_request.user_id,  # type: ignore[union-attr]
            doc_ids=doc_ids,
            reranker=reranker,
            # Always hybrid; top_k reads from VECTOR_SEARCH_TOP_K / KEYWORD_SEARCH_TOP_K env vars
        )
        if lf:
            lf.update_current_span(output={"chunks_returned": len(raw_chunks)})
    except Exception:
        logger.warning("agent_search_failed", extra={"entity": entity})
        if lf:
            lf.update_current_span(output={"chunks_returned": 0, "error": True})
        AGENT_TOOL_CALLS.labels("search_documents", "error").inc()
        AGENT_TOOL_DURATION.labels("search_documents").observe(perf_counter() - _tool_started)
        return _SearchResult(
            entity=entity,
            chunks=[],
            payloads={},
            error_str=f"Search failed for entity: {entity}",
            rewrite_stats=rewrite_stats,
        )
    finally:
        _search_lf_stack.close()

    chunks = [dc_replace(c, turn_index=iteration) for c in raw_chunks]
    payloads = await get_chunk_prompt_payloads(session, [c.chunk_id for c in chunks])
    AGENT_TOOL_CALLS.labels("search_documents", "ok").inc()
    AGENT_TOOL_DURATION.labels("search_documents").observe(perf_counter() - _tool_started)
    return _SearchResult(
        entity=entity, chunks=chunks, payloads=payloads, rewrite_stats=rewrite_stats
    )


# ---------------------------------------------------------------------------
# Finalizer handling
# ---------------------------------------------------------------------------


def _resolve_candidate_refs(
    candidate: AgentFindings | AnalyticalFindings,
    state: AgentRunState,
    request_id: str,
) -> AgentFindings | AnalyticalFindings:
    """Rewrite source_chunks / evidence_chunks / refuted_by S-labels into chunk UUIDs.

    Unresolvable refs are dropped (never propagated downstream — a leaked label would
    surface in the synthesis prompt as a citable ID that has no matching excerpt).
    """
    all_unresolved: list[str] = []
    result: AgentFindings | AnalyticalFindings

    if isinstance(candidate, AgentFindings):
        new_findings = []
        for f in candidate.findings:
            resolved, unresolved = state.evidence.resolve_refs(f.source_chunks)
            all_unresolved.extend(unresolved)
            new_findings.append(f.model_copy(update={"source_chunks": resolved}))
        result = candidate.model_copy(update={"findings": tuple(new_findings)})
    else:
        new_obs = []
        for o in candidate.observations:
            evidence, unresolved = state.evidence.resolve_refs(o.evidence_chunks)
            all_unresolved.extend(unresolved)
            refuted: list[str] | None = o.refuted_by
            if o.refuted_by is not None:
                refuted, unresolved = state.evidence.resolve_refs(o.refuted_by)
                all_unresolved.extend(unresolved)
            new_obs.append(
                o.model_copy(update={"evidence_chunks": evidence, "refuted_by": refuted})
            )
        result = candidate.model_copy(update={"observations": tuple(new_obs)})

    if all_unresolved:
        logger.warning(
            "agent_chunk_refs_unresolved",
            extra={"request_id": request_id, "unresolved_refs": all_unresolved},
        )
    return result


def _log_finalizer_accept(lf, candidate: AgentFindings | AnalyticalFindings) -> None:
    if not lf:
        return
    with contextlib.suppress(Exception):
        if isinstance(candidate, AgentFindings):
            findings_summary = [
                {
                    "entity": f.entity,
                    "available": f.available,
                    "value": f.value,
                    "currency": f.currency,
                    "unit": f.unit,
                    "period_end": f.period_end,
                    "source_chunks": f.source_chunks,
                    "reason": f.reason,
                }
                for f in candidate.findings
            ]
            lf.update_current_span(
                output={
                    "metric_requested": candidate.metric_requested,
                    "comparison_op": candidate.comparison_op,
                    "findings": findings_summary,
                },
                metadata={"parse_ok": True, "findings_count": len(candidate.findings)},
            )
        else:
            lf.update_current_span(
                output={
                    "question": candidate.question,
                    "conclusion": candidate.conclusion,
                    "gaps": candidate.gaps,
                    "observations": [
                        {
                            "claim": o.claim,
                            "confidence": o.confidence,
                            "evidence_chunks": o.evidence_chunks,
                        }
                        for o in candidate.observations
                    ],
                },
                metadata={"parse_ok": True, "observations_count": len(candidate.observations)},
            )


async def _handle_finalizer(
    finalizer_tc: ToolCallRef,
    state: AgentRunState,
    redis_app: Redis,
    request_id: str,
) -> TurnOutcome:
    lf = lf_client.get_client()
    _fin_lf_stack = contextlib.ExitStack()
    if lf:
        try:
            fin_input = json.loads(finalizer_tc.arguments)
        except Exception:
            fin_input = {"raw": finalizer_tc.arguments[:500]}
        _fin_lf_stack.enter_context(
            lf.start_as_current_observation(as_type="span", name=finalizer_tc.name, input=fin_input)
        )
    try:
        try:
            raw_candidate = gates_module.parse_findings(finalizer_tc)
        except ValidationError:
            AGENT_TOOL_CALLS.labels(finalizer_tc.name, "error").inc()
            logger.warning(
                "agent_findings_parse_failed",
                extra={"request_id": request_id, "raw_args": finalizer_tc.arguments[:500]},
            )
            if lf:
                lf.update_current_span(
                    level="ERROR",
                    metadata={"parse_ok": False, "raw_args": finalizer_tc.arguments[:300]},
                )
            await add_event(
                redis_app,
                request_id,
                "tool_call_completed",
                {"entity": "__finalizer__", "error": True, "reason": "findings_parse_failed"},
            )
            return Stop("natural")

        candidate = _resolve_candidate_refs(raw_candidate, state, request_id)
        if isinstance(candidate, AnalyticalFindings):
            candidate = gates_module.drop_evidence_free_observations(candidate)
        # Protect chunks the model cited here even if this attempt is rejected — a
        # later accepted call must still be able to cite them.
        state.evidence.protect(gates_module.finding_chunk_ids(candidate))
        # Fold every finalizer attempt (accepted or rejected below) into the ledger —
        # this is what gives a non-converged run real content to serve (degraded) and
        # exercises the in-place revision path across retried attempts. C6's grounding
        # filter lives in `record`, so ungrounded items are dropped, not admitted.
        state.findings.ingest(candidate, state.evidence)

        for gate in tools_module.gates_for(finalizer_tc.name):
            reason = gate(candidate, state)
            if reason is not None:
                await gates_module.reject(
                    reason=reason,
                    finalizer_tc=finalizer_tc,
                    candidate=candidate,
                    state=state,
                    redis_app=redis_app,
                    request_id=request_id,
                )
                return Continue()

        AGENT_TOOL_CALLS.labels(finalizer_tc.name, "ok").inc()
        state.sealed = True
        _log_finalizer_accept(lf, candidate)
        return Finalize(candidate)
    finally:
        _fin_lf_stack.close()


# ---------------------------------------------------------------------------
# One turn
# ---------------------------------------------------------------------------


async def _run_turn(
    state: AgentRunState,
    llm: RoutedLLM,
    tools: list[dict],
    chat_state: ChatPipelineState,
    session: AsyncSession,
    session_factory: async_sessionmaker[AsyncSession],
    reranker: Reranker | None,
    redis_app: Redis,
    request_id: str,
    is_analytical: bool,
    search_sem: asyncio.Semaphore,
    execute_search: ExecuteSearchFn,
    rewrite_model_id: str,
    max_chunks_per_lookup: int,
) -> TurnOutcome:
    iteration = state.iteration
    await add_event(redis_app, request_id, "agent_turn_started", {"iteration": iteration})
    logger.debug("agent_turn_started", extra={"request_id": request_id, "iteration": iteration})

    lf = lf_client.get_client()
    _turn_lf_stack = contextlib.ExitStack()
    if lf:
        _turn_lf_stack.enter_context(
            lf.start_as_current_observation(
                as_type="span",
                name=f"agent_turn_{iteration}",
                input={
                    "iteration": iteration,
                    "messages": [
                        {
                            "role": m.role.value,
                            "content": (m.content or "")[:500],
                            "tool_call_id": m.tool_call_id,
                            "tool_calls": (
                                [
                                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                                    for tc in m.tool_calls
                                ]
                                if m.tool_calls
                                else None
                            ),
                        }
                        for m in state.transcript.messages
                    ],
                },
            )
        )

    new_chunks = 0
    turn: AssistantTurnResult | None = None
    try:
        turn = await asyncio.wait_for(
            llm.complete_with_tools(state.transcript.messages, tools=tools, temperature=0.0),
            timeout=state.turn_timeout_seconds,
        )
        if turn.stats:
            state.record_spend(llm.model_id, turn.stats)
            if turn.stats.input_tokens:
                LLM_TOKENS.labels("input", llm.model_id).inc(turn.stats.input_tokens)
            if turn.stats.output_tokens:
                LLM_TOKENS.labels("output", llm.model_id).inc(turn.stats.output_tokens)
            if turn.stats.cached_input_tokens:
                LLM_CACHE_HIT_TOKENS.labels(llm.model_id).inc(turn.stats.cached_input_tokens)
            if turn.stats.cost_usd:
                LLM_COST.labels(llm.model_id).inc(turn.stats.cost_usd)

            if chat_state.llm_request and chat_state.llm_request.conversation_id is not None:
                with contextlib.suppress(Exception):
                    await LLMRequestRepository(session).create_subrequest(
                        parent_request_id=chat_state.llm_request.id,
                        conversation_id=chat_state.llm_request.conversation_id,
                        user_id=chat_state.llm_request.user_id,
                        provider=llm.provider,
                        model=llm.model_id,
                        request_type="agent_tool_call",
                        request_params={
                            "iteration": iteration,
                            "tool_calls_issued": len(turn.tool_calls or []),
                        },
                        status="completed",
                        **stats_to_request_kwargs(turn.stats),
                    )

        if not turn.tool_calls:
            return Stop("natural")

        finalizer_tc = next(
            (tc for tc in turn.tool_calls if tools_module.is_terminal(tc.name)), None
        )
        if finalizer_tc is not None:
            return await _handle_finalizer(finalizer_tc, state, redis_app, request_id)

        search_tcs = turn.tool_calls
        state.tool_calls_total += len(search_tcs)
        state.transcript.append_tool_calls(search_tcs)

        async def _guarded_search(tc: ToolCallRef) -> _SearchResult:
            async with search_sem, session_factory() as task_session:
                # A fresh session per concurrent search — the shared `session` is not
                # safe for concurrent use under asyncio.gather (P0-1).
                return await execute_search(
                    tc, chat_state, task_session, reranker, redis_app, request_id, iteration
                )

        results = await asyncio.gather(*[_guarded_search(tc) for tc in search_tcs])

        for tc, result in zip(search_tcs, results, strict=False):
            if result.rewrite_stats:
                state.record_spend(rewrite_model_id, result.rewrite_stats)
            lookup_id = state.evidence.start_lookup()
            entity_new = state.evidence.admit(lookup_id, iteration, result.chunks)
            new_chunks += entity_new
            if result.entity:
                state.searched_entities.add(result.entity)
            logger.debug(
                "tool_call_completed",
                extra={
                    "request_id": request_id,
                    "iteration": iteration,
                    "entity": result.entity,
                    "chunks_returned": len(result.chunks),
                    "new_chunks_added": entity_new,
                },
            )
            await add_event(
                redis_app,
                request_id,
                "tool_call_completed",
                {
                    "entity": result.entity,
                    "chunks_returned": len(result.chunks),
                    "new_chunks_added": entity_new,
                },
            )
            # Assemble the tool-result context here (sequentially) so S-labels
            # continue across searches instead of restarting at S1 each time.
            if result.error_str is not None:
                tool_content = result.error_str
            else:
                # P2 (audit finding): admit the full result above for provenance
                # (seen_in_lookups / P1-5), but render only the top-N into the
                # transcript — mid-loop, uncapped tool results were the single
                # biggest per-turn token cost (~106k chars measured). The record
                # stays complete; only the view is capped. `apply_cap` (post-loop)
                # now exists purely as the final synthesis-selection safety net.
                ctx = state.evidence.assign_labels(
                    result.chunks[:max_chunks_per_lookup], result.payloads
                )
                tool_content = ctx.formatted_context or "(no results)"
            state.transcript.append(
                ChatMessage(role=Role.tool, tool_call_id=tc.id, content=tool_content)
            )

        if new_chunks == 0:
            # For extraction/comparison, an empty round means the retrievable surface
            # is exhausted — stop. For analytical queries, it means "this query found
            # nothing new," which is a reason to search *differently*, not to finalize
            # on thin evidence (doc Pattern 3, sub-task 3.i). Allow a bounded number of
            # empty rounds to reformulate before falling back to convergence.
            state.empty_rounds += 1
            if not is_analytical or state.empty_rounds > state.effort.max_empty_rounds:
                return Stop("convergence")
            state.transcript.append(
                ChatMessage(
                    role=Role.user,
                    content=(
                        "That search returned no new evidence. Do not finalize yet — the "
                        "drivers you need are likely in a different section (a footnote, "
                        "reconciliation, or segment table). Reformulate search_documents with "
                        "different terms targeting where the magnitudes are disclosed."
                    ),
                )
            )
            state.transcript.compress()
            return Continue()

        # P0-2: a productive round (new chunks admitted) resets the empty-round streak —
        # `max_empty_rounds` counts *consecutive* empty rounds, not cumulative.
        state.empty_rounds = 0

        if not state.spend_within_budget():
            return Stop("budget_cap")

        state.transcript.compress()
        return Continue()
    finally:
        if lf:
            lf.update_current_span(
                output={
                    "tool_calls": [
                        {"name": tc.name, "arguments": tc.arguments}
                        for tc in (turn.tool_calls if turn is not None and turn.tool_calls else [])
                    ],
                },
                metadata={
                    "token_spend_cumulative": state.input_tokens_total(),
                    "new_chunks": new_chunks,
                },
            )
        _turn_lf_stack.close()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def run_loop(
    chat_state: ChatPipelineState,
    llm: RoutedLLM,
    session: AsyncSession,
    redis_app: Redis,
    request_id: str,
    reranker: Reranker | None,
    session_factory: async_sessionmaker[AsyncSession],
    *,
    execute_search: ExecuteSearchFn = _execute_search,
) -> tuple[dict[UUID, RetrievedChunk], AgentFindings | AnalyticalFindings | None, AgentLoopMeta]:
    """Run the agent tool-calling loop for retrieval queries.

    Returns (chunk_registry, agent_findings, meta). chunk_registry is keyed by
    chunk_id; values have turn_index stamped. agent_findings is the FindingsLedger
    projection — the accumulated findings, marked degraded when no finalizer sealed the
    run, and None only when no finalizer was ever attempted.

    ``session`` is used for the loop's own serial DB work (subrequest logging).
    Concurrent searches each open their own session from ``session_factory``:
    SQLAlchemy's AsyncSession is not safe for concurrent use, so the fan-out under
    asyncio.gather must never share one (P0-1).
    """
    settings = get_agent_settings()

    query_shape = (
        chat_state.router_output.query_shape  # type: ignore[union-attr]
        if chat_state.router_output and hasattr(chat_state.router_output, "query_shape")
        else None
    )
    is_analytical = query_shape == "analytical"
    tools = tools_module.ALL_TOOLS
    prompt_name = "v3_agent_analytical" if is_analytical else "v3_agent"
    effort = EffortPrior.for_shape(settings, query_shape)
    search_sem = asyncio.Semaphore(effort.max_concurrent_searches)
    rewrite_model_id = get_query_transformer_model()

    system_content = get_system_prompt(version=prompt_name)
    # context_messages always ends with the current-turn user message (loaded with
    # before_seq=assistant_seq, which includes it) — drop it here since it's appended
    # explicitly below via chat_state.user_query_raw (post prompt-injection sanitization).
    history = (chat_state.context_messages or [])[:-1]
    messages: list[ChatMessage] = [
        ChatMessage(role=Role.system, content=system_content),
        *[ChatMessage(role=Role(m.role.value), content=m.content or "") for m in history],
    ]

    # Inject exact entity names from the router so the agent uses correct strings and
    # knows which entities it must cover before calling report_findings. Also inject
    # metadata-backed years so the agent does not hallucinate fiscal year terms.
    _entity_years: dict[str, list[int]] = {}
    if chat_state.scope_result and chat_state.scope_result.entity_manifest:
        for item in chat_state.scope_result.entity_manifest:
            years = sorted(
                {s["year"] for s in (item.doc_summaries or []) if s.get("year")},
                reverse=True,
            )
            if years:
                _entity_years[item.entity_name] = years

    # Seeded regardless of query_shape: both finalizers are always in the tool pool
    # now (Stage 1.5), so missing_entity_gate — which only fires for report_findings
    # (isinstance-scoped, see gates.py) — must have real coverage data whichever
    # finalizer the model ends up calling, not just for non-analytical shapes.
    expected_entities: set[str] = set()
    if chat_state.scope_result and chat_state.scope_result.per_entity_doc_ids:
        expected_entities = set(chat_state.scope_result.per_entity_doc_ids.keys())

    if not is_analytical and expected_entities:
        lines: list[str] = []
        for name in sorted(expected_entities):
            years = _entity_years.get(name)
            suffix = f" (available years: {', '.join(str(y) for y in years)})" if years else ""
            lines.append(f"- {name}{suffix}")
        messages.append(
            ChatMessage(
                role=Role.user,
                content=(
                    "Entities to search (you MUST call search_documents for each before report_findings).\n"
                    "Use ONLY the listed years in your search queries — do not guess or invent fiscal years:\n"
                    + "\n".join(lines)
                ),
            )
        )
    elif is_analytical and _entity_years:
        year_lines = [
            f"- {name}: {', '.join(str(y) for y in years)}"
            for name, years in sorted(_entity_years.items())
        ]
        messages.append(
            ChatMessage(
                role=Role.user,
                content=(
                    "Available document years (use ONLY these in search queries — do not invent fiscal years):\n"
                    + "\n".join(year_lines)
                ),
            )
        )

    messages.append(ChatMessage(role=Role.user, content=chat_state.user_query_raw))

    state = AgentRunState(
        effort=effort,
        token_budget=settings.token_budget,
        turn_timeout_seconds=settings.turn_timeout_seconds,
        transcript=Transcript(messages),
        expected_entities=expected_entities,
    )

    iterations_run = 0
    for iteration in range(effort.max_iterations):
        state.iteration = iteration
        iterations_run = iteration + 1
        try:
            outcome = await _run_turn(
                state,
                llm,
                tools,
                chat_state,
                session,
                session_factory,
                reranker,
                redis_app,
                request_id,
                is_analytical,
                search_sem,
                execute_search,
                rewrite_model_id,
                settings.max_chunks_per_entity,
            )
        except TimeoutError:
            logger.warning(
                "agent_turn_timeout",
                extra={
                    "request_id": request_id,
                    "iteration": iteration,
                    "timeout": state.turn_timeout_seconds,
                },
            )
            state.convergence_reason = "timeout"
            break

        if isinstance(outcome, Finalize):
            state.convergence_reason = "natural"
            break
        if isinstance(outcome, Stop):
            state.convergence_reason = outcome.reason
            break
        # Continue(): fall through to the next iteration

    # Post-loop, once: per-lookup context-window cap (never evicts a protected chunk).
    state.evidence.apply_cap(settings.max_chunks_per_entity)

    await add_event(
        redis_app,
        request_id,
        "agent_synthesis_starting",
        {"total_chunks": len(state.evidence), "iterations": iterations_run},
    )

    meta = build_meta(state, iterations_run)
    # Serve the ledger projection, not a single accepted candidate: a non-converged run
    # now yields its accumulated partial findings marked degraded, instead of None →
    # raw-excerpt fallback (P1-6). None only when no finalizer was ever attempted.
    findings = state.findings.projection(degraded=not state.sealed)
    return state.evidence.registry, findings, meta
