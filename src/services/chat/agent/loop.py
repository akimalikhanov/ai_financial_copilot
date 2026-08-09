"""The agent tool-calling loop.

Turn control flow is explicit: `_run_turn` returns a `TurnOutcome` (`Continue` or `Stop`)
instead of the old ~8 scattered `convergence_reason = ...; break` sites threaded through
one function.

Post-D3 there is one termination model and no terminal tool. Reports are incremental —
`_apply_report` folds each into the `FindingsLedger` and returns a tool result — and the
*loop* decides when the run is done, by checking plan coverage. That deletes the whole
rejection subsystem (gates, `reject`, stubbed calls, retry budgets): nothing is ever
un-done, so nothing has to be re-attempted, and the coercion-to-restate that forced the
revert of `2d43ed8` has no reason to exist.
"""

from __future__ import annotations

import asyncio
import contextlib
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
from src.observability.langfuse import span as lf_span
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
from src.services.chat.agent import tools as tools_module
from src.services.chat.agent.evidence import EvidenceLedger
from src.services.chat.agent.findings import Candidate, drop_evidence_free_observations
from src.services.chat.agent.state import (
    AgentLoopMeta,
    AgentRunState,
    AspectStats,
    ConvergenceReason,
    EffortPrior,
    build_meta,
    debug_snapshot,
    get_agent_settings,
    open_aspects,
    render_status,
    turn_snapshot,
)
from src.services.chat.agent.tools import SearchDocumentsArgs
from src.services.chat.agent.transcript import Transcript, cap_history
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
from src.services.security.injection_detector import scan_user_input
from src.utils.config import get_injection_scan_user_input_enabled, get_query_transformer_model

if TYPE_CHECKING:
    from src.schemas.chat import ChatMessage as SchemaChatMessage
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
    # D6 must not conflate "the corpus was unreachable" with "the model sent bad
    # arguments" — only the former justifies Stop("search_unavailable") or a
    # couldn't-search gap. A malformed tool call is the model's problem, not the backend's.
    backend_failed: bool = False


ExecuteSearchFn = Callable[
    [ToolCallRef, "ChatPipelineState", AsyncSession, "Reranker | None", Redis, str, int, bool],
    Awaitable[_SearchResult],
]


# ---------------------------------------------------------------------------
# Turn outcome
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Continue:
    pass


@dataclass(frozen=True)
class Stop:
    reason: ConvergenceReason


# No `Finalize`: post-D3 no tool call ends the run. The loop decides, so every exit is a
# `Stop` with a reason — including `covered`, which is what a successful run now looks like.
TurnOutcome = Continue | Stop


# ---------------------------------------------------------------------------
# Search execution
# ---------------------------------------------------------------------------


def _mint(plan: dict[str, str], sub_question: str | None, max_plan_items: int) -> str | None:
    """Assign the aspect id for one search, minting a new one when the sub_question is new.

    The loop mints rather than the model: a model-authored slug is a string-equality join
    across turns on free text, and a small model writing `input_costs` on turn 1 and
    `input_cost_inflation` on turn 3 lands the record under a phantom key while the real
    aspect stays open forever. Copying a two-character token echoed back in the tool
    result is a far easier instruction to follow.

    Exact-match-after-normalization only — a near-duplicate mints a new id. The cost is one
    redundant plan entry, bounded by `max_plan_items`; the alternative is a similarity
    threshold with no defensible value. Returns None for a blank or over-cap sub_question:
    those searches still execute, but untracked, so per-aspect stats under-report them.
    """
    q = " ".join((sub_question or "").split())
    if not q:
        return None
    norm = q.lower()
    for aid, existing in plan.items():
        if " ".join(existing.lower().split()) == norm:
            return aid
    if len(plan) >= max_plan_items:
        return None
    aid = f"A{len(plan) + 1}"
    plan[aid] = q
    return aid


def _sub_question_of(tc: ToolCallRef) -> str | None:
    """The search call's `sub_question`, or None if absent/malformed.

    Deliberately tolerant: a call whose arguments don't parse still executes (and fails
    with its own error downstream), it just mints no aspect.
    """
    try:
        return SearchDocumentsArgs.model_validate_json(tc.arguments).sub_question
    except ValidationError:
        return None


async def _execute_search(
    tc: ToolCallRef,
    state: ChatPipelineState,
    session: AsyncSession,
    reranker: Reranker | None,
    redis_app: Redis,
    request_id: str,
    iteration: int,
    is_analytical: bool,
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
    raw_query = search_args.query

    # Resolve doc_ids for this entity. The analytical agent passes entity="" — resolve it
    # to the primary entity's name here so the ledger, the SSE events and the trace span
    # all carry the entity actually searched rather than an empty string.
    per_entity = (state.scope_result.per_entity_doc_ids or {}) if state.scope_result else {}
    entity = search_args.entity
    if entity and entity in per_entity:
        doc_ids = per_entity[entity]
    elif not entity and per_entity:
        # Scope to the first (primary) entity's docs rather than leaking to the full corpus.
        entity, doc_ids = next(iter(per_entity.items()))
    else:
        doc_ids = state.scope_result.doc_ids if state.scope_result else None

    _tool_started = perf_counter()
    await add_event(redis_app, request_id, "tool_call_started", {"entity": entity})

    # Rewrite at tool boundary — cheap model, eval-independent. Skipped on the analytical
    # path (10b step 5): there the tool model composes a targeted, hypothesis-shaped query
    # per aspect, so the rewriter is a second model second-guessing it — it blurs specific
    # causal terms into generic finance vocabulary, and costs 3-5 calls per turn on the
    # critical path. BM25 loses its differentiated keyword_query; acceptable because the
    # v5 prompt asks for keyword-dense queries without the company name, and retrieval is
    # already scoped to this entity's doc_ids so a leaked entity term has ~zero IDF.
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

    with lf_span(
        f"tool_search_{entity}_{iteration}",
        as_type="retriever",
        input={
            "entity": entity,
            "query": raw_query,
            # Show the resolved scope this search was constrained to, so the
            # trace makes clear which docs the agent could actually see.
            "scope_doc_ids": [str(d) for d in doc_ids] if doc_ids else "all",
            "scope_doc_count": len(doc_ids) if doc_ids else "all",
            "scoped_via_entity": entity in per_entity,
        },
    ) as obs:
        rewrite_stats: LLMResponseStats | None = None
        if is_analytical:
            transformed = TransformedQuery(
                semantic_query=raw_query, keyword_query=raw_query, fallback=False
            )
        else:
            try:
                transformed, rewrite_stats = await rewrite_query(
                    raw_query,
                    scope_docs=scope_docs or None,
                    session=session,
                    parent_request_id=state.llm_request.id if state.llm_request else None,
                    conversation_id=state.conversation_id,
                    user_id=state.llm_request.user_id if state.llm_request else None,
                    extra_request_params={
                        "entity": entity,
                        "iteration": iteration,
                        "source": "agent",
                    },
                )
            except Exception:
                logger.warning("agent_rewrite_failed", extra={"entity": entity, "query": raw_query})
                transformed = TransformedQuery(
                    semantic_query=raw_query,
                    keyword_query=raw_query,
                    fallback=True,
                )
        try:
            _, retrieval_trace, raw_chunks = await run_chat_rag_pipeline(
                session,
                transformed=transformed,
                user_id=state.llm_request.user_id,  # type: ignore[union-attr]
                doc_ids=doc_ids,
                reranker=reranker,
                # Always hybrid; top_k reads from VECTOR_SEARCH_TOP_K / KEYWORD_SEARCH_TOP_K env vars
            )
            if obs:
                obs.update(output={"chunks_returned": len(raw_chunks)})
            # A total backend outage fails open inside the pipeline (empty results, no
            # exception), so it reaches here looking exactly like "the corpus has nothing
            # on this". Only the trace can tell the two apart.
            if retrieval_trace.all_backends_failed:
                logger.warning("agent_search_backends_down", extra={"entity": entity})
                if obs:
                    obs.update(output={"chunks_returned": 0, "error": True})
                AGENT_TOOL_CALLS.labels("search_documents", "error").inc()
                AGENT_TOOL_DURATION.labels("search_documents").observe(
                    perf_counter() - _tool_started
                )
                return _SearchResult(
                    entity=entity,
                    chunks=[],
                    payloads={},
                    error_str=f"Search failed for entity: {entity}",
                    rewrite_stats=rewrite_stats,
                    backend_failed=True,
                )
        except Exception:
            logger.warning("agent_search_failed", extra={"entity": entity})
            if obs:
                obs.update(output={"chunks_returned": 0, "error": True})
            AGENT_TOOL_CALLS.labels("search_documents", "error").inc()
            AGENT_TOOL_DURATION.labels("search_documents").observe(perf_counter() - _tool_started)
            return _SearchResult(
                entity=entity,
                chunks=[],
                payloads={},
                error_str=f"Search failed for entity: {entity}",
                rewrite_stats=rewrite_stats,
                backend_failed=True,
            )

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


def _parse_report(tc: ToolCallRef) -> Candidate:
    """Parse a report tool call against its Pydantic schema.

    Raises ``pydantic.ValidationError`` on malformed JSON or a schema violation, surfaced
    at the caller instead of silently constructing garbage via ``.get()``.
    """
    if tc.name == "report_findings":
        return AgentFindings.model_validate_json(tc.arguments)
    return AnalyticalFindings.model_validate_json(tc.arguments)


def _candidate_keys(candidate: Candidate) -> set[str]:
    if isinstance(candidate, AgentFindings):
        return {f.entity for f in candidate.findings}
    return {o.aspect for o in candidate.observations}


def _filter_to_keys(candidate: Candidate, keys: set[str]) -> Candidate:
    if isinstance(candidate, AgentFindings):
        return candidate.model_copy(
            update={"findings": tuple(f for f in candidate.findings if f.entity in keys)}
        )
    return candidate.model_copy(
        update={"observations": tuple(o for o in candidate.observations if o.aspect in keys)}
    )


def _analytical_insufficiency(findings: AnalyticalFindings) -> str | None:
    """Advisory text, never a gate (D3).

    Post-D3 there is no rejection to attach this to: it is appended to the report result
    so the model can act on it, and ignoring it costs nothing but a weaker answer. Its one
    surviving clause carries the real sufficiency signal — self-reported gaps are
    deliberately *not* penalized (10b §1e), since firing on honesty trains the model to
    stop reporting gaps at all.
    """
    # Stated negatives are excluded, not counted as low-confidence: nudging a re-search on
    # an aspect the model has already settled as absent is the same "firing on honesty"
    # this advisory exists to avoid, and `confidence` is meaningless on a negative.
    substantiated = [o for o in findings.observations if o.substantiated]
    if not substantiated:
        return None
    if all(o.confidence == "low" for o in substantiated):
        return (
            "Every observation so far is low-confidence. Search differently — likely a "
            "footnote, reconciliation, or segment table — to corroborate."
        )
    return None


def _render_report_result(state: AgentRunState, closed: set[str], unknown: set[str]) -> str:
    """The tool result for one report: what landed, what didn't, and what is still open.

    Deliberately echoes only *this turn's* change plus the open list. That open list goes
    stale as the run proceeds, which is acceptable because the status view (appended last
    on every subsequent call) is the single source of coverage truth and wins by recency.
    """
    parts: list[str] = []
    if closed:
        parts.append(f"Recorded {', '.join(sorted(closed))}.")
    if unknown:
        plural = len(unknown) != 1
        parts.append(
            f"{', '.join(repr(k) for k in sorted(unknown))} "
            f"{'are' if plural else 'is'} not an open key and "
            f"{'were' if plural else 'was'} not recorded. "
            "Use the key shown in brackets in the search result, or issue search_documents "
            "with a new sub_question first."
        )
    if not parts:
        parts.append("Nothing was recorded — no known key was reported.")
    still_open = open_aspects(state)
    if still_open:
        parts.append("Open: " + ", ".join(f"{a} ({state.plan[a]})" for a in still_open))
    else:
        parts.append("All planned items are now reported.")
    projection = state.findings.projection()
    if isinstance(projection, AnalyticalFindings):
        advisory = _analytical_insufficiency(projection)
        if advisory:
            parts.append(advisory)
    return " ".join(parts)


def _apply_report(tc: ToolCallRef, state: AgentRunState, request_id: str) -> str:
    """Fold one report into the ledger and return its tool result.

    Partial acceptance, never rejection (D3): unknown keys are dropped from the candidate
    and named in the result; known ones land. Nothing is un-done, so there is no rejection
    path, no stub and no restatement to coerce — which is precisely why the coverage
    pressure `2d43ed8` had to revert is safe to apply here.
    """
    state.report_calls_total += 1
    if state.turns_to_first_report is None:
        state.turns_to_first_report = state.iteration

    try:
        parsed = _parse_report(tc)
    except ValidationError:
        AGENT_TOOL_CALLS.labels(tc.name, "error").inc()
        logger.warning(
            "agent_report_parse_failed",
            extra={"request_id": request_id, "raw_args": tc.arguments[:500]},
        )
        # A parse failure is a tool result, not the end of the run: non-terminal means the
        # model gets told and retries, where today a malformed finalizer ends the run.
        return (
            f"{tc.name} arguments did not parse against the schema. "
            "Re-issue the call with valid arguments."
        )

    candidate = _resolve_candidate_refs(parsed, state, request_id)
    raw_keys = _candidate_keys(candidate)

    # Plan keys are loop-minted (extraction seeds them from expected_entities), so a key
    # the loop never minted has no referent — it is named back rather than recorded.
    known = raw_keys & state.plan.keys()
    unknown = raw_keys - state.plan.keys()
    if unknown:
        state.unknown_aspect_keys += len(unknown)
        candidate = _filter_to_keys(candidate, known)

    # Closed from the *raw* keys, before the grounding filter: a key the documents
    # genuinely don't answer must close, or the loop hammers it to budget death. Grounded
    # → a finding; ungrounded → a gap. Both close it, neither closes it silently (D4
    # derives `addressed` from findings ∪ closed_as_gap).
    state.reported_keys |= known
    if isinstance(candidate, AnalyticalFindings):
        before = len(candidate.observations)
        candidate = drop_evidence_free_observations(candidate)
        state.ungrounded_closes += before - len(candidate.observations)
    state.findings.ingest(candidate, state.evidence)

    # D4 reconciliation: anything reported but neither grounded nor gapped would otherwise
    # close silently, serving a key that produced no output at all.
    for key in sorted(known & state.unaccounted_keys()):
        state.findings.add_gap(
            f"Reported but unsupported by retrieved evidence: {state.plan[key]}", closes=key
        )

    AGENT_TOOL_CALLS.labels(tc.name, "ok").inc()
    return _render_report_result(state, closed=known, unknown=unknown)


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
) -> TurnOutcome:
    iteration = state.iteration
    await add_event(redis_app, request_id, "agent_turn_started", {"iteration": iteration})
    logger.debug("agent_turn_started", extra={"request_id": request_id, "iteration": iteration})

    # The status view is computed per call and appended last — never stored, so it never
    # invalidates the cached prefix and never accumulates. It is the single source of
    # coverage truth: nothing permanent in the transcript states coverage. It also doubles
    # as this turn's trace input: the transcript up to here is what earlier turns' own
    # spans already recorded (Langfuse nests them under the same agent_loop chain), so
    # re-dumping all of `state.transcript.messages` on every turn would repeat turn 0's
    # content N times by turn N for no new information.
    status = render_status(state)
    with lf_span(
        f"agent_turn_{iteration}",
        input={
            "iteration": iteration,
            "message_count": len(state.transcript.messages),
            "status": status,
        },
    ) as obs:
        return await _run_turn_inner(
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
            iteration,
            status,
            obs,
        )


async def _run_turn_inner(
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
    iteration: int,
    status: str | None,
    obs: object,
) -> TurnOutcome:
    """Body of one turn, run inside `_run_turn`'s `agent_turn_{N}` span.

    Split out only so the span in `_run_turn` can wrap it with a plain `with` (the span
    needs `status` computed before it opens, to use as trimmed trace input instead of the
    full transcript replay) while this keeps the turn's own try/finally for tool-call output.
    """
    new_chunks = 0
    turn: AssistantTurnResult | None = None
    results_by_id: dict[str, str] = {}
    try:
        prompt_messages = [
            *state.transcript.messages,
            *([ChatMessage(role=Role.user, content=status)] if status else []),
        ]
        turn = await asyncio.wait_for(
            llm.complete_with_tools(prompt_messages, tools=tools, temperature=0.0),
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
            # With no terminal tool this is a normal exit, not a rare one: the model
            # emitted prose instead of a call. projection() still serves whatever the
            # ledger accumulated (None only if nothing was ever reported).
            return Stop("natural")

        reports = [tc for tc in turn.tool_calls if tc.name in tools_module.REPORT_TOOL_NAMES]
        searches = [tc for tc in turn.tool_calls if tc.name not in tools_module.REPORT_TOOL_NAMES]

        # Rule 1: the assistant message is appended ONCE, verbatim, in emission order.
        # Rule 2 (below): exactly one role=tool result per tool_call id. An assistant
        # tool_calls entry without a matching result — or vice versa — is a 400 on every
        # OpenAI-compatible provider. Both are easy to honour now that no branch executes
        # some calls and discards others (the bug at the old loop.py:520-524).
        state.transcript.append_tool_calls(turn.tool_calls)
        state.tool_calls_total += len(turn.tool_calls)

        # Mint before execution, so the tool result can echo the id the model must cite.
        minted: dict[str, str | None] = {
            tc.id: _mint(state.plan, _sub_question_of(tc), state.effort.max_plan_items)
            for tc in searches
        }

        async def _guarded_search(tc: ToolCallRef) -> _SearchResult:
            async with search_sem, session_factory() as task_session:
                # A fresh session per concurrent search — the shared `session` is not
                # safe for concurrent use under asyncio.gather (P0-1).
                return await execute_search(
                    tc,
                    chat_state,
                    task_session,
                    reranker,
                    redis_app,
                    request_id,
                    iteration,
                    is_analytical,
                )

        results: list[_SearchResult] = []
        if searches:
            results = list(await asyncio.gather(*[_guarded_search(tc) for tc in searches]))

        backend_failures = 0
        for tc, result in zip(searches, results, strict=False):
            if result.rewrite_stats:
                state.record_spend(rewrite_model_id, result.rewrite_stats)
            entity_new = state.evidence.admit(result.chunks)
            new_chunks += entity_new
            if result.entity:
                state.searched_entities.add(result.entity)

            # D5: per-aspect search provenance, written at the one instant everything is
            # in hand. A failed or empty search admits no chunks, so this cannot be
            # reconstructed from the EvidenceLedger afterwards — which is exactly the
            # case D6 has to tell apart.
            aspect = minted.get(tc.id)
            if aspect is not None:
                stats = state.aspect_stats.setdefault(aspect, AspectStats())
                stats.searches += 1
                stats.new_chunks += entity_new
                if result.backend_failed:
                    stats.errored += 1
            if result.backend_failed:
                backend_failures += 1

            logger.debug(
                "tool_call_completed",
                extra={
                    "request_id": request_id,
                    "iteration": iteration,
                    "entity": result.entity,
                    "aspect": aspect,
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
            # Assemble the tool-result context here (sequentially) so S-labels continue
            # across searches instead of restarting at S1 each time.
            if result.error_str is not None:
                results_by_id[tc.id] = result.error_str
            else:
                # P2 (audit finding): admit the full result above for provenance, but
                # render only the top-N into the transcript — mid-loop, uncapped tool
                # results were the single biggest per-turn token cost (~106k chars
                # measured). The record stays complete; only the view is capped.
                ctx = state.evidence.assign_labels(
                    result.chunks[: state.effort.max_chunks_per_lookup],
                    result.payloads,
                    max_revivals=state.effort.max_revivals_per_turn,
                )
                body = ctx.formatted_context or "(no results)"
                # The aspect id is echoed back so "reuse the key in brackets" is a copy
                # from adjacent context, not a slug reconstructed from memory.
                results_by_id[tc.id] = f"[{aspect}] {body}" if aspect else body

        # Reports fold after searches but read nothing from them: the model composed these
        # before seeing this turn's results, so it cannot cite them.
        before_addressed = set(state.addressed)
        for tc in reports:
            results_by_id[tc.id] = _apply_report(tc, state, request_id)
        closed_this_turn = state.addressed - before_addressed

        # Rule 2: exactly one result per call id, in turn.tool_calls order.
        for tc in turn.tool_calls:
            state.transcript.append(
                ChatMessage(
                    role=Role.tool,
                    tool_call_id=tc.id,
                    content=results_by_id.get(tc.id, "(no result)"),
                )
            )

        # --- termination, loop-owned (step 3) ---
        # Ordering is load-bearing: minting happened before this, so a turn that closes the
        # last open key *and* opens a new thread continues rather than stopping. Today the
        # same emission stops the run and discards the model's own statement that a thread
        # remains open.
        if state.plan and not open_aspects(state):
            state.sealed_by_coverage = True
            return Stop("covered")

        # A dead backend is not an empty corpus. Today it produces new_chunks == 0 and the
        # model is told to "reformulate with different terms" while burning its budget.
        if searches and backend_failures == len(searches):
            return Stop("search_unavailable")

        # Progress = new chunks *or* a key closed. A turn that settles a key from evidence
        # already in hand is real progress; without this, resolving the last aspects from
        # admitted evidence trips convergence one turn before coverage — the most likely
        # spurious stop in this design.
        if new_chunks == 0 and not closed_this_turn:
            state.empty_rounds += 1
            if not is_analytical or state.empty_rounds > state.effort.max_empty_rounds:
                return Stop("convergence")
            # The stall nudge is no longer appended here: a permanent user message that
            # accumulates one copy per empty round (10b §4, row 8). It moved into
            # `render_status`, which is recomputed per call and keyed off `empty_rounds`.
        else:
            # `max_empty_rounds` counts *consecutive* empty rounds, not cumulative (P0-2).
            state.empty_rounds = 0

        if not state.spend_within_budget():
            return Stop("budget_cap")
        if state.past_deadline():
            return Stop("deadline")

        state.transcript.compress(state.evidence)
        return Continue()
    finally:
        if obs:
            obs.update(  # type: ignore[attr-defined]
                output={
                    "tool_calls": [
                        {"name": tc.name, "arguments": tc.arguments}
                        for tc in (turn.tool_calls if turn is not None and turn.tool_calls else [])
                    ],
                    # The `role: tool` results this turn actually appended to the
                    # transcript, keyed by tool_call_id — what the *next* turn's model
                    # call will read back. Without this, the only place a search's
                    # rendered excerpts or a report's fold-in result are visible is
                    # buried inside the next turn's full llm.complete_with_tools
                    # GENERATION input (present, but not this span's — nothing on
                    # agent_turn_N itself showed what this turn actually produced).
                    "tool_results": results_by_id,
                    # Structured coverage state *after* this turn's effects landed — the
                    # per-turn counterpart to the prose `status` string the model saw as
                    # this span's input (which reflects state *before* the turn ran).
                    "coverage_after": turn_snapshot(state),
                },
                metadata={
                    "token_spend_cumulative": state.input_tokens_total(),
                    "new_chunks": new_chunks,
                },
            )


CARRYOVER_STUB = "[prior turn: restated earlier results in a different format]"


def build_agent_history(
    history: list[SchemaChatMessage],
    *,
    scan: bool,
    request_id: str = "",
) -> list[ChatMessage]:
    """Project conversation history into the agent's transcript.

    Contract F1: only role and content cross. `findings_block` cannot reach the agent —
    ChatMessage here is the frozen/slotted adapter type with no such field — and an
    answer derived from a carried block is stubbed, since its prose restates numbers the
    agent has no evidence for.
    """
    out: list[ChatMessage] = []
    for m in history:
        content = m.content or ""
        if m.role.value == "assistant" and m.answer_derived_from_carryover:
            content = CARRYOVER_STUB
        elif scan and m.role.value == "user" and content:
            signal = scan_user_input(content)
            if signal.severity == "block":
                logger.info(
                    "agent_history_turn_blocked",
                    extra={"request_id": request_id, "matched_rules": signal.matched_rules},
                )
                continue
            content = signal.sanitized_text
        out.append(ChatMessage(role=Role(m.role.value), content=content))
    return out


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
) -> tuple[
    EvidenceLedger,
    AgentFindings | AnalyticalFindings | None,
    AgentLoopMeta,
]:
    """Run the agent tool-calling loop for retrieval queries.

    Returns (evidence, agent_findings, meta). The ledger carries the ordered chunks, the
    payloads cached at render time (so synthesis hydrates nothing, D2), and the rendered
    subset — what the model could actually read, the only defensible pool for synthesis to
    fall back on. agent_findings is the FindingsLedger projection — the accumulated
    findings, marked degraded when no finalizer sealed the run, and None only when no
    finalizer was ever attempted.

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
    # One branch, prompt and pool on the same line: a prompt naming a tool the model was
    # not given is a broken run, so nothing may assign one without the other.
    prompt_name, tools = (
        ("v5_agent_analytical", tools_module.ANALYTICAL_TOOLS)
        if is_analytical
        else ("v3_agent", tools_module.EXTRACTION_TOOLS)
    )
    effort = EffortPrior.for_shape(settings, query_shape)
    search_sem = asyncio.Semaphore(effort.max_concurrent_searches)
    rewrite_model_id = get_query_transformer_model()

    system_content = get_system_prompt(version=prompt_name)
    # context_messages always ends with the current-turn user message (loaded with
    # before_seq=assistant_seq, which includes it) — drop it here since it's appended
    # explicitly below via chat_state.user_query_raw (post prompt-injection sanitization).
    history = (chat_state.context_messages or [])[:-1]
    # Prior user turns reach the agent unsanitized: `history.append_user` writes the raw
    # `req.content` at the API layer, while `scan_user_input` runs later in the worker and
    # rewrites only the *current* turn. So a prior turn that scored "block" — one the
    # pipeline refused to answer — still lands in the agent transcript verbatim, because
    # it was written to the chat tail before the task ran. Scan them here (10b §4d):
    # strip invisibles/role markers as the current turn gets, and drop a blocked turn
    # outright rather than handing the agent the exact text the guardrail rejected.
    scan = get_injection_scan_user_input_enabled()
    history_messages = build_agent_history(history, scan=scan, request_id=request_id)
    messages: list[ChatMessage] = [
        ChatMessage(role=Role.system, content=system_content),
        # Row 2 of 10b §4 was the largest unbounded cost in the turn: up to 50 prior
        # messages, permanent, dwarfing the excerpts they contextualize.
        *cap_history(
            history_messages,
            max_turns=settings.history_turns,
            max_assistant_chars=settings.history_assistant_tokens * 4,
        ),
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

    # Seeded regardless of query_shape: the entity-injection message below and
    # `_inject_unsearched_stubs` at the synthesis boundary both read this to tell
    # "searched and found nothing" from "never searched", whichever finalizer runs.
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
        deadline_seconds=settings.deadline_seconds,
        transcript=Transcript(messages),
        expected_entities=expected_entities,
    )

    # D3: one termination model. Extraction's plan is loop-authored — seeded from the
    # entities the router resolved, keyed by entity name so the model reports under the
    # name it was given. `missing_entity_gate` computed exactly this set as a rejection
    # reason; now it is the same coverage check the analytical path uses. The analytical
    # plan seeds itself from turn-1 search `sub_question`s instead.
    if not is_analytical:
        for name in sorted(expected_entities):
            state.plan[name] = name

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

        if isinstance(outcome, Stop):
            state.convergence_reason = outcome.reason
            break
        # Continue(): fall through to the next iteration

    # Step 8: an aspect that never closed becomes a stated limitation rather than
    # vanishing. `_render_observations_block` already renders gaps as "Unresolved (do not
    # assert as fact)", so there is no synthesis-side change. Ordered before the degraded
    # caveat so a non-converged run reads "here is specifically what is missing" rather
    # than the generic caveat alone. Depends on §1c: without the gaps-union fix a later
    # report with gaps=[] would wipe these before they are read.
    # Snapshot coverage *before* the step-8 gaps below, which mark every remaining key
    # addressed. Reading it afterwards would score an unresolved key as covered and make
    # plan_covered/plan_seeded — step 9's kill criterion — permanently 1.0.
    state.plan_covered_at_stop = len(state.plan.keys() & state.addressed)

    for aspect in open_aspects(state):
        stats = state.aspect_stats.get(aspect)
        # D6: a permanently-open aspect whose every search errored is a backend failure,
        # not an absence in the corpus. Saying "not in the documents" there would be
        # confidently, silently wrong.
        if stats is not None and stats.searches > 0 and stats.errored == stats.searches:
            gap = f"Could not be checked — document search was unavailable: {state.plan[aspect]}"
        else:
            gap = f"Not resolved: {state.plan[aspect]}"
        # Only the analytical envelope has a `gaps` field to carry these. On the extraction
        # path an unreported entity is already surfaced by `_inject_unsearched_stubs`, and
        # forcing a kind here would make an all-failed run project the wrong shape.
        state.findings.add_gap(gap, closes=aspect, establishes_kind=is_analytical)

    await add_event(
        redis_app,
        request_id,
        "agent_synthesis_starting",
        {"total_chunks": len(state.evidence), "iterations": iterations_run},
    )

    meta = build_meta(state, iterations_run)
    # Terminal state, attached to the enclosing "agent_loop" chain span (opened by the
    # caller, current here since no per-turn span is open past the loop) — otherwise a
    # non-converged/degraded run's final ledger contents are only reconstructable by
    # replaying every tool-call span in order.
    lf = lf_client.get_client()
    if lf:
        with contextlib.suppress(Exception):
            lf.update_current_span(metadata={"final_state": debug_snapshot(state)})
    # Serve the ledger projection, not a single accepted candidate: a non-converged run
    # now yields its accumulated partial findings marked degraded, instead of None →
    # raw-excerpt fallback (P1-6). None only when no finalizer was ever attempted.
    findings = state.findings.projection(degraded=not state.sealed)
    return state.evidence, findings, meta
