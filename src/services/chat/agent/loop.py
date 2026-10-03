"""The agent tool-calling loop.

Turn control flow is explicit: `_run_turn` returns a `TurnOutcome` (`Continue` or `Stop`).

There is one termination model and no terminal tool. Reports are incremental —
`_apply_report` folds each into the `FindingsLedger` and returns a tool result — and the
*loop* decides when the run is done, by checking plan coverage. Nothing a report lands is
ever un-done, so there is no rejection path and nothing has to be re-attempted or restated.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable, Sequence
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
    AGENT_TOOL_ARG_ERRORS,
    AGENT_TOOL_CALLS,
    AGENT_TOOL_DURATION,
    LLM_CACHE_HIT_TOKENS,
    LLM_COST,
    LLM_TOKENS,
    observe_llm_latency,
)
from src.redis_client import add_event
from src.repository.llm_request_repository import LLMRequestRepository, stats_to_request_kwargs
from src.schemas.agent_findings import AgentFindings, AnalyticalFindings
from src.schemas.query_transform import ScopeDocSummary, TransformedQuery
from src.schemas.retrieval import ChunkPromptPayload, RetrievedChunk
from src.services.chat.agent import tools as tools_module
from src.services.chat.agent.evidence import EvidenceLedger
from src.services.chat.agent.findings import Candidate
from src.services.chat.agent.state import (
    AgentLoopMeta,
    AgentRunState,
    AspectStats,
    ConvergenceReason,
    build_meta,
    get_agent_settings,
    open_aspects,
    render_status,
    snapshot,
    unresolved_lines,
)
from src.services.chat.agent.tools import SearchDocumentsArgs
from src.services.chat.agent.transcript import Transcript, cap_history
from src.services.chat.events import build_activity_event
from src.services.llm_adapters.base_adapter import (
    AssistantTurnResult,
    ChatMessage,
    LLMResponseStats,
    Role,
    ToolCallRef,
)
from src.services.llm_runtime.exceptions import LLMError
from src.services.prompts.prompt_renderer import get_system_prompt
from src.services.retrieval.chat_rag import run_chat_rag_pipeline
from src.services.retrieval.payload_hydrator import get_chunk_prompt_payloads
from src.services.retrieval.query_transformer import rewrite_query
from src.services.security.injection_detector import scan_user_input
from src.utils.config import get_injection_scan_user_input_enabled, get_query_transformer_model

if TYPE_CHECKING:
    from src.schemas.chat import ChatMessage as SchemaChatMessage
    from src.schemas.chat import ChatPipelineState
    from src.services.llm_router import LLMRouter, RoutedLLM
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
    # Never conflate "the corpus was unreachable" with "the model sent bad
    # arguments" — only the former justifies Stop("search_unavailable") or a
    # couldn't-search gap. A malformed tool call is the model's problem, not the backend's.
    backend_failed: bool = False
    args_invalid: bool = False
    # Capabilities this search ran without ("dense", "keyword", "rerank"). Partial
    # degradation, as opposed to backend_failed's total outage: the search still returned
    # usable chunks, just from fewer sources than it should have.
    degraded: frozenset[str] = frozenset()
    # False when this search's chunk scores are fusion scores rather than cross-encoder
    # ones — true whenever reranking fell open *or* is switched off. The two scales are an
    # order of magnitude apart, so confidence thresholds must not be applied to them.
    scores_are_rerank: bool = True
    # The `tool_call_started` activity event's id, so completion correlates by id
    # rather than by entity name — None when no started event was ever emitted
    # (invalid tool-call arguments, resolved before entity/id assignment).
    activity_id: str | None = None


ExecuteSearchFn = Callable[
    [ToolCallRef, "ChatPipelineState", AsyncSession, "Reranker | None", Redis, str, int, bool],
    Awaitable[_SearchResult],
]


@dataclass(frozen=True)
class RunDeps:
    """Everything a turn needs that is fixed for the whole run."""

    # The tool model first, then its fallbacks. A provider error moves a turn to the next.
    llms: tuple[RoutedLLM, ...]
    chat_state: ChatPipelineState
    # The loop's own serial DB work (sub-request logging). Concurrent searches each open
    # their own session from `session_factory` instead.
    session: AsyncSession
    session_factory: async_sessionmaker[AsyncSession]
    reranker: Reranker | None
    redis_app: Redis
    request_id: str
    is_analytical: bool
    search_sem: asyncio.Semaphore
    execute_search: ExecuteSearchFn
    rewrite_model_id: str


# ---------------------------------------------------------------------------
# Turn outcome
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Continue:
    pass


@dataclass(frozen=True)
class Stop:
    reason: ConvergenceReason


# No tool call ends the run. The loop decides, so every exit is a `Stop` with a reason —
# including `covered`, which is what a successful run looks like.
TurnOutcome = Continue | Stop


@dataclass(frozen=True)
class TurnFacts:
    """What one turn's tool calls did — all `decide` needs beyond the run state."""

    searches: int
    backend_failures: int
    new_chunks: int
    closed: frozenset[str]  # plan keys that produced their first finding this turn


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
        AGENT_TOOL_ARG_ERRORS.labels("search_documents").inc()
        return _SearchResult(
            entity="",
            chunks=[],
            payloads={},
            error_str="search_documents call had invalid arguments — entity and query are required strings.",
            args_invalid=True,
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
    activity_id, start_data = build_activity_event(
        "tool_call_started",
        label=entity,
        parent_id=f"round-{iteration}",
        detail={"tool": "search"},
    )
    await add_event(redis_app, request_id, "activity", start_data)

    # Rewrite at tool boundary — cheap model, eval-independent. Skipped on the analytical
    # path: there the tool model composes a targeted, hypothesis-shaped query
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
                    activity_id=activity_id,
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
                activity_id=activity_id,
            )

    degraded = frozenset(
        name
        for name, ok in (
            ("dense", retrieval_trace.embed_ok and retrieval_trace.vector_ok),
            ("keyword", retrieval_trace.keyword_ok),
            ("rerank", retrieval_trace.rerank_ok),
        )
        if not ok
    )
    if degraded:
        logger.warning(
            "agent_search_degraded", extra={"entity": entity, "degraded": sorted(degraded)}
        )

    chunks = [dc_replace(c, turn_index=iteration) for c in raw_chunks]
    payloads = await get_chunk_prompt_payloads(session, [c.chunk_id for c in chunks])
    AGENT_TOOL_CALLS.labels("search_documents", "ok").inc()
    AGENT_TOOL_DURATION.labels("search_documents").observe(perf_counter() - _tool_started)
    return _SearchResult(
        entity=entity,
        chunks=chunks,
        payloads=payloads,
        rewrite_stats=rewrite_stats,
        degraded=degraded,
        scores_are_rerank=retrieval_trace.scores_are_rerank,
        activity_id=activity_id,
    )


# ---------------------------------------------------------------------------
# Report handling
# ---------------------------------------------------------------------------


def _resolve_candidate_refs(
    candidate: AgentFindings | AnalyticalFindings,
    state: AgentRunState,
    request_id: str,
) -> AgentFindings | AnalyticalFindings:
    """Rewrite source_chunks / evidence_chunks S-labels into chunk UUIDs.

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
            new_obs.append(o.model_copy(update={"evidence_chunks": evidence}))
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


def _searched(state: AgentRunState, key: str) -> bool:
    stats = state.aspect_stats.get(key)
    return key in state.searched_entities or (stats is not None and stats.searches > 0)


def _drop_unsearched_negatives(
    candidate: Candidate, state: AgentRunState
) -> tuple[Candidate, set[str]]:
    """The candidate without stated negatives for keys no search has covered, and those
    keys. A negative closes its key with nothing cited, so it must at least follow a search
    for that key."""
    if isinstance(candidate, AgentFindings):
        dropped = {
            f.entity
            for f in candidate.findings
            if not f.available and not _searched(state, f.entity)
        }
        kept_findings = tuple(
            f for f in candidate.findings if f.available or f.entity not in dropped
        )
        return candidate.model_copy(update={"findings": kept_findings}), dropped
    dropped = {
        o.aspect
        for o in candidate.observations
        if not o.substantiated and not _searched(state, o.aspect)
    }
    kept_observations = tuple(
        o for o in candidate.observations if o.substantiated or o.aspect not in dropped
    )
    return candidate.model_copy(update={"observations": kept_observations}), dropped


def _render_report_result(
    state: AgentRunState, closed: set[str], unknown: set[str], unsearched: set[str]
) -> str:
    """The tool result for one report: what landed, what didn't, and what is still open.

    Deliberately echoes only *this turn's* change plus the open list. That open list goes
    stale as the run proceeds, which is acceptable because the status view (appended last
    on every subsequent call) is the single source of coverage truth and wins by recency.
    """
    parts: list[str] = []
    # Split on what actually landed, not on what was claimed: with no mid-loop gap
    # reconciliation an ungrounded report leaves its key open, and telling the model it
    # was "recorded" would be the one message that stops it retrying the aspect.
    addressed = state.addressed
    landed = sorted(closed & addressed)
    dropped = sorted(closed - addressed - unsearched)
    if landed:
        parts.append(f"Recorded {', '.join(landed)}.")
    if unsearched - addressed:
        keys = sorted(unsearched - addressed)
        plural = len(keys) != 1
        parts.append(
            f"{', '.join(keys)} {'were' if plural else 'was'} not recorded as absent — no "
            f"earlier search covered {'them' if plural else 'it'}. Search first, then report "
            f"what the results show."
        )
    if dropped:
        plural = len(dropped) != 1
        parts.append(
            f"{', '.join(dropped)} {'were' if plural else 'was'} not recorded — "
            f"{'their' if plural else 'its'} observations cited no chunk from the evidence "
            f"you retrieved. Re-report citing chunk labels from a search result, or, if the "
            f"documents do not support the aspect, report it with substantiated: false."
        )
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
    return " ".join(parts)


def _apply_report(tc: ToolCallRef, state: AgentRunState, request_id: str) -> str:
    """Fold one report into the ledger and return its tool result.

    Partial acceptance, never rejection: unknown keys are dropped from the candidate and
    named in the result; known ones land. Nothing is un-done, so there is no rejection
    path, no stub and no restatement to coerce.
    """
    state.report_calls_total += 1
    if state.turns_to_first_report is None:
        state.turns_to_first_report = state.iteration

    try:
        parsed = _parse_report(tc)
    except ValidationError:
        AGENT_TOOL_CALLS.labels(tc.name, "error").inc()
        AGENT_TOOL_ARG_ERRORS.labels(tc.name).inc()
        state.report_parse_failures += 1
        logger.warning(
            "agent_report_parse_failed",
            extra={"request_id": request_id, "raw_args": tc.arguments[:500]},
        )
        # A parse failure is a tool result, not the end of the run: the model is told and
        # retries.
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
    candidate, unsearched = _drop_unsearched_negatives(candidate, state)
    state.unsearched_negatives += len(unsearched)

    # A reported-but-ungrounded key stays open: closing it would let `Stop("covered")`
    # seal a run that produced no grounded output for that aspect. Left open, it stays
    # searchable, and if it is still open at the end projection renders it as unresolved.
    state.findings.ingest(candidate, state.evidence)

    AGENT_TOOL_CALLS.labels(tc.name, "ok").inc()
    return _render_report_result(state, closed=known, unknown=unknown, unsearched=unsearched)


# ---------------------------------------------------------------------------
# One turn
# ---------------------------------------------------------------------------


def fold_searches(
    state: AgentRunState,
    searches: list[ToolCallRef],
    results: list[_SearchResult],
    minted: dict[str, str | None],
    rewrite_model_id: str,
) -> tuple[dict[str, str], list[int]]:
    """Fold one turn's search results into the state, in call order.

    Returns the tool-result text per call id and the count of newly admitted chunks per
    search. No awaits and no I/O: labels are assigned here, sequentially, so S-labels
    continue across searches instead of restarting at S1, and stay deterministic however
    the concurrent searches finished.
    """
    texts: dict[str, str] = {}
    new_per_search: list[int] = []
    for tc, result in zip(searches, results, strict=True):
        if result.rewrite_stats:
            state.record_spend(rewrite_model_id, result.rewrite_stats)
        entity_new = state.evidence.admit(result.chunks)
        new_per_search.append(entity_new)
        if result.entity:
            state.searched_entities.add(result.entity)
        state.degraded_capabilities |= result.degraded
        if result.chunks and not result.scores_are_rerank:
            state.scores_are_rerank = False
        if result.args_invalid:
            state.search_arg_errors += 1

        # Per-aspect search provenance, written at the one instant everything is in
        # hand. A failed or empty search admits no chunks, so this cannot be
        # reconstructed from the EvidenceLedger afterwards — and `unresolved_lines`
        # needs it to tell "backend down" from "not in the documents".
        aspect = minted.get(tc.id)
        if aspect is not None:
            stats = state.aspect_stats.setdefault(aspect, AspectStats())
            stats.searches += 1
            stats.new_chunks += entity_new
            if result.backend_failed:
                stats.errored += 1

        if result.error_str is not None:
            texts[tc.id] = result.error_str
            continue
        # Admit the full result above for provenance, but render only the top-N into the
        # transcript — uncapped tool results are the biggest per-turn token cost. The
        # record stays complete; only the view is capped.
        ctx = state.evidence.assign_labels(
            result.chunks[: state.settings.max_chunks_per_entity],
            result.payloads,
            max_revivals=state.settings.max_revivals_per_turn,
        )
        body = ctx.formatted_context or "(no results)"
        # The aspect id is echoed back so "reuse the key in brackets" is a copy from
        # adjacent context, not a slug reconstructed from memory.
        texts[tc.id] = f"[{aspect}] {body}" if aspect else body
    return texts, new_per_search


def fold_reports(
    state: AgentRunState, reports: list[ToolCallRef], request_id: str
) -> tuple[dict[str, str], frozenset[str]]:
    """Fold one turn's reports into the ledger, in call order.

    Returns the tool-result text per call id and the plan keys this turn closed. Called
    after minting and before this turn's searches run: the model wrote these reports
    without seeing those results, so they are judged against the state the previous turn
    left — they cite only earlier labels, and a negative needs an earlier search.
    """
    before = set(state.addressed)
    texts = {tc.id: _apply_report(tc, state, request_id) for tc in reports}
    return texts, frozenset(state.addressed - before)


def decide(state: AgentRunState, facts: TurnFacts) -> TurnOutcome:
    """Whether the run stops after this turn, and why.

    Called once the turn's tool results are in the transcript. Writes only the two fields
    the decision owns, `sealed_by_coverage` and `empty_rounds`.
    """
    # Ordering is load-bearing: minting happened before this, so a turn that closes the
    # last open key *and* opens a new thread continues rather than stopping.
    if state.plan and not open_aspects(state):
        state.sealed_by_coverage = True
        return Stop("covered")

    # A dead backend is not an empty corpus: without this stop it produces
    # new_chunks == 0 and the model is told to reformulate while burning its budget.
    if facts.searches and facts.backend_failures == facts.searches:
        return Stop("search_unavailable")

    # Progress = new chunks *or* a key closed. A turn that settles a key from evidence
    # already in hand is real progress; without this, resolving the last aspects from
    # admitted evidence trips convergence one turn before coverage.
    if facts.new_chunks == 0 and not facts.closed:
        state.empty_rounds += 1
        if state.empty_rounds > state.settings.max_empty_rounds:
            return Stop("convergence")
        # The stall nudge lives in `render_status`, recomputed per call from
        # `empty_rounds`, so it never accumulates in the transcript.
    else:
        # `max_empty_rounds` counts *consecutive* empty rounds, not cumulative.
        state.empty_rounds = 0

    if not state.spend_within_budget():
        return Stop("budget_cap")
    return Continue()


async def _call_tool_model(
    state: AgentRunState, deps: RunDeps, messages: list[ChatMessage], tools: list[dict]
) -> tuple[RoutedLLM, AssistantTurnResult]:
    """The first model in the chain that answers, and its turn. The last model's provider
    error propagates; a timeout does not fall back."""

    async def complete(llm: RoutedLLM) -> AssistantTurnResult:
        return await asyncio.wait_for(
            llm.complete_with_tools(messages, tools=tools, temperature=0.0),
            timeout=state.settings.turn_timeout_seconds,
        )

    *fallible, last = deps.llms
    for llm in fallible:
        try:
            return llm, await complete(llm)
        except LLMError as e:
            logger.warning(
                "agent_tool_model_fallback",
                extra={
                    "request_id": deps.request_id,
                    "iteration": state.iteration,
                    "from_model": llm.model_id,
                    "error": type(e).__name__,
                },
            )
    return last, await complete(last)


async def _finish_despite_cancel(aw: Awaitable[None]) -> None:
    """Run `aw` to completion even if the run deadline cancels this task meanwhile, then
    let the cancellation through. A commit interrupted on the shared session would leave it
    unusable for the pipeline's own writes after the loop."""
    task = asyncio.ensure_future(aw)
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise


async def _record_turn_spend(
    state: AgentRunState, deps: RunDeps, llm: RoutedLLM, turn: AssistantTurnResult
) -> None:
    """Spend, metrics and the `llm_requests` sub-request row for one tool-model call."""
    stats = turn.stats
    if stats is None:
        return
    state.record_spend(llm.model_id, stats)
    if stats.input_tokens:
        LLM_TOKENS.labels("input", llm.model_id).inc(stats.input_tokens)
    if stats.output_tokens:
        LLM_TOKENS.labels("output", llm.model_id).inc(stats.output_tokens)
    if stats.cached_input_tokens:
        LLM_CACHE_HIT_TOKENS.labels(llm.model_id).inc(stats.cached_input_tokens)
    if stats.cost_usd:
        LLM_COST.labels(llm.model_id).inc(stats.cost_usd)
    observe_llm_latency(llm.model_id, "agent_tool_call", stats)

    llm_request = deps.chat_state.llm_request
    if llm_request is None or llm_request.conversation_id is None:
        return
    with contextlib.suppress(Exception):
        await LLMRequestRepository(deps.session).create_subrequest(
            parent_request_id=llm_request.id,
            conversation_id=llm_request.conversation_id,
            user_id=llm_request.user_id,
            provider=llm.provider,
            model=llm.model_id,
            request_type="agent_tool_call",
            request_params={
                "iteration": state.iteration,
                "tool_calls_issued": len(turn.tool_calls or []),
            },
            status="completed",
            **stats_to_request_kwargs(stats),
        )
        # Release the pgbouncer server connection between turns. create_subrequest only
        # flushes, so without this the transaction it opens stays open across the next
        # turn's LLM call — converting transaction pooling into session pooling for the
        # whole loop. Committed here and not inside create_subrequest because naming.py's
        # caller depends on NOT committing: its sub-request and the title update have to
        # land together.
        await deps.session.commit()


async def _guarded_search(tc: ToolCallRef, state: AgentRunState, deps: RunDeps) -> _SearchResult:
    """One search, bounded by the turn timeout; a timeout becomes a backend failure."""
    try:
        async with asyncio.timeout(state.settings.turn_timeout_seconds):
            # A fresh session per concurrent search — the shared `deps.session` is not
            # safe for concurrent use under asyncio.gather.
            async with deps.search_sem, deps.session_factory() as task_session:
                return await deps.execute_search(
                    tc,
                    deps.chat_state,
                    task_session,
                    deps.reranker,
                    deps.redis_app,
                    deps.request_id,
                    state.iteration,
                    deps.is_analytical,
                )
    except TimeoutError:
        logger.warning(
            "agent_search_timeout",
            extra={
                "request_id": deps.request_id,
                "iteration": state.iteration,
                "timeout_s": state.settings.turn_timeout_seconds,
            },
        )
        AGENT_TOOL_CALLS.labels("search_documents", "error").inc()
        return _SearchResult(
            entity="",
            chunks=[],
            payloads={},
            error_str="search timed out — the search backend did not respond in time.",
            backend_failed=True,
        )


async def _run_turn(state: AgentRunState, deps: RunDeps, tools: list[dict]) -> TurnOutcome:
    """One turn: call the tool model, dispatch its calls, fold the results, decide."""
    iteration = state.iteration
    request_id = deps.request_id
    _, round_data = build_activity_event(
        "round_started",
        event_id=f"round-{iteration}",
        label=f"Round {iteration + 1}",
        detail={"iteration": iteration},
    )
    await add_event(deps.redis_app, request_id, "activity", round_data)
    logger.debug("agent_turn_started", extra={"request_id": request_id, "iteration": iteration})

    # The status view is computed per call and appended last — never stored, so it never
    # invalidates the cached prefix and never accumulates. It is the single source of
    # coverage truth: nothing permanent in the transcript states coverage. It also doubles
    # as this turn's trace input: the transcript up to here is what earlier turns' own
    # spans already recorded (Langfuse nests them under the same agent_loop chain), so
    # re-dumping all of `state.transcript.messages` on every turn would repeat turn 0's
    # content N times by turn N for no new information.
    status = render_status(state)
    new_chunks = 0
    turn: AssistantTurnResult | None = None
    results_by_id: dict[str, str] = {}
    with lf_span(
        f"agent_turn_{iteration}",
        input={
            "iteration": iteration,
            "message_count": len(state.transcript.messages),
            "status": status,
        },
    ) as obs:
        try:
            prompt_messages = [
                *state.transcript.messages,
                *([ChatMessage(role=Role.user, content=status)] if status else []),
            ]
            served, turn = await _call_tool_model(state, deps, prompt_messages, tools)
            await _finish_despite_cancel(_record_turn_spend(state, deps, served, turn))

            if not turn.tool_calls:
                # With no terminal tool this is a normal exit, not a rare one: the model
                # emitted prose instead of a call. projection() still serves whatever the
                # ledger accumulated (None only if nothing was ever reported).
                return Stop("natural")

            # A call to a tool outside this turn's pool is answered and never parsed: an
            # extraction run cannot land an Observation, and a final-turn search does not
            # run.
            offered = {t["function"]["name"] for t in tools}
            report_names = offered & tools_module.REPORT_TOOL_NAMES
            reports = [tc for tc in turn.tool_calls if tc.name in report_names]
            searches = [tc for tc in turn.tool_calls if tc.name in offered - report_names]
            for tc in turn.tool_calls:
                if tc.name not in offered:
                    logger.warning(
                        "agent_tool_not_available",
                        extra={"request_id": request_id, "tool": tc.name},
                    )
                    results_by_id[tc.id] = (
                        f"Tool {tc.name!r} is not available. "
                        f"Available: {', '.join(sorted(offered))}."
                    )

            # The assistant message is appended ONCE, verbatim, in emission order, and
            # (below) there is exactly one role=tool result per tool_call id. An assistant
            # tool_calls entry without a matching result — or vice versa — is a 400 on
            # every OpenAI-compatible provider.
            state.transcript.append_tool_calls(turn.tool_calls)
            state.tool_calls_total += len(turn.tool_calls)

            # Mint before execution, so the tool result can echo the id the model must cite.
            minted: dict[str, str | None] = {
                tc.id: _mint(state.plan, _sub_question_of(tc), state.settings.max_plan_items)
                for tc in searches
            }

            report_texts, closed = fold_reports(state, reports, request_id)
            results_by_id |= report_texts
            for _ in reports:
                report_id, report_start = build_activity_event(
                    "tool_call_started",
                    label="Recording findings",
                    parent_id=f"round-{iteration}",
                    detail={"tool": "report"},
                )
                await add_event(deps.redis_app, request_id, "activity", report_start)
                _, report_end = build_activity_event("tool_call_ended", event_id=report_id)
                await add_event(deps.redis_app, request_id, "activity", report_end)

            results = list(
                await asyncio.gather(*[_guarded_search(tc, state, deps) for tc in searches])
            )
            search_texts, new_per_search = fold_searches(
                state, searches, results, minted, deps.rewrite_model_id
            )
            results_by_id |= search_texts
            new_chunks = sum(new_per_search)
            for tc, result, entity_new in zip(searches, results, new_per_search, strict=True):
                logger.debug(
                    "tool_call_completed",
                    extra={
                        "request_id": request_id,
                        "iteration": iteration,
                        "entity": result.entity,
                        "aspect": minted.get(tc.id),
                        "chunks_returned": len(result.chunks),
                        "new_chunks_added": entity_new,
                    },
                )
                if result.activity_id is not None:
                    _, end_data = build_activity_event(
                        "tool_call_ended",
                        event_id=result.activity_id,
                        detail={
                            "chunks_returned": len(result.chunks),
                            "new_chunks_added": entity_new,
                        },
                    )
                    await add_event(deps.redis_app, request_id, "activity", end_data)

            # Exactly one result per call id, in turn.tool_calls order.
            for tc in turn.tool_calls:
                state.transcript.append(
                    ChatMessage(
                        role=Role.tool,
                        tool_call_id=tc.id,
                        content=results_by_id.get(tc.id, "(no result)"),
                    )
                )

            outcome = decide(
                state,
                TurnFacts(
                    searches=len(searches),
                    backend_failures=sum(r.backend_failed for r in results),
                    new_chunks=new_chunks,
                    closed=closed,
                ),
            )
            if isinstance(outcome, Continue):
                state.transcript.compress(state.evidence)
            return outcome
        finally:
            if obs:
                obs.update(
                    output={
                        "tool_calls": [
                            {"name": tc.name, "arguments": tc.arguments}
                            for tc in (turn.tool_calls if turn and turn.tool_calls else [])
                        ],
                        # The `role: tool` results this turn actually appended to the
                        # transcript, keyed by tool_call_id — what the *next* turn's model
                        # call will read back. Without this, a search's rendered excerpts or
                        # a report's fold-in result are visible only inside the next turn's
                        # full GENERATION input, not on this span.
                        "tool_results": results_by_id,
                        # State *after* this turn's effects landed — the structured
                        # counterpart to the prose `status` this span took as input (state
                        # *before* the turn ran).
                        "state_after": snapshot(state),
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

    Only role and content cross. `findings_block` cannot reach the agent —
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


def prompt_and_tools(query_shape: str | None) -> tuple[str, list[dict]]:
    """The tool-model prompt and tool pool for a shape.

    Returned together because a prompt naming a tool the model was not given is a broken
    run. Public so the pipeline can tag traces with the prompt before the loop starts.
    """
    if query_shape == "analytical":
        return "v5_agent_analytical", tools_module.ANALYTICAL_TOOLS
    return "v3_agent", tools_module.EXTRACTION_TOOLS


def tool_model_chain(router: LLMRouter, model_id: str) -> list[RoutedLLM]:
    """The tool model followed by its configured fallback, keeping only models that can
    call tools. Raises when the tool model itself cannot."""
    chain = router.get_with_fallback(model_id)
    if not chain[0].capabilities.get("tool_calling", False):
        raise RuntimeError(
            f"AGENT_TOOL_MODEL={model_id!r} does not have tool_calling: true in models.yaml"
        )
    return [m for m in chain if m.capabilities.get("tool_calling", False)]


async def _iterate(state: AgentRunState, deps: RunDeps, tools: list[dict]) -> None:
    """Run turns until one stops the run or the iteration cap is reached; records the stop
    reason on `state`."""
    request_id = deps.request_id
    for iteration in range(state.max_iterations):
        state.iteration = iteration
        # An aspect whose evidence is already admitted but never written up dies as
        # "Not resolved" at the iteration cap. Withholding search on the last turn — the
        # turn that was going to run anyway — gives the model one pass to convert what it
        # already holds, at no extra cost. `tool_choice` stays "auto": if the model emits
        # prose instead, the turn returns Stop("natural") and accumulated findings still
        # serve, so this fails safe.
        final_turn = iteration == state.max_iterations - 1
        turn_tools = (
            [t for t in tools if t["function"]["name"] in tools_module.REPORT_TOOL_NAMES]
            if final_turn
            else tools
        )
        try:
            outcome = await _run_turn(state, deps, turn_tools)
        except TimeoutError:
            logger.warning(
                "agent_turn_timeout",
                extra={
                    "request_id": request_id,
                    "iteration": iteration,
                    "timeout": state.settings.turn_timeout_seconds,
                },
            )
            state.convergence_reason = "timeout"
            return
        except LLMError as e:
            # Earlier turns' evidence and findings are already in the ledgers, so a late
            # provider error serves them degraded. With neither, there is nothing to serve.
            if len(state.evidence) == 0 and not state.findings.keys():
                raise
            logger.warning(
                "agent_tool_model_error",
                extra={
                    "request_id": request_id,
                    "iteration": iteration,
                    "error": type(e).__name__,
                },
            )
            state.convergence_reason = "llm_error"
            return

        if isinstance(outcome, Stop):
            state.convergence_reason = outcome.reason
            return


async def run_loop(
    chat_state: ChatPipelineState,
    llm: RoutedLLM,
    session: AsyncSession,
    redis_app: Redis,
    request_id: str,
    reranker: Reranker | None,
    session_factory: async_sessionmaker[AsyncSession],
    *,
    fallbacks: Sequence[RoutedLLM] = (),
    execute_search: ExecuteSearchFn = _execute_search,
) -> tuple[
    EvidenceLedger,
    AgentFindings | AnalyticalFindings | None,
    AgentLoopMeta,
]:
    """Run the agent tool-calling loop for retrieval queries.

    Returns (evidence, agent_findings, meta). The ledger carries the ordered chunks, the
    payloads cached at render time (so synthesis hydrates nothing), and the rendered
    subset — what the model could actually read, the only defensible pool for synthesis to
    fall back on. agent_findings is the FindingsLedger projection — the accumulated
    findings, marked degraded when the run did not seal, and None only when no report
    was ever attempted.

    ``fallbacks`` are tried in order when ``llm`` raises a provider error. When the whole
    chain fails, the run stops as ``llm_error`` and serves what it gathered; with nothing
    gathered the error propagates and the request fails.

    ``session`` is used for the loop's own serial DB work (subrequest logging).
    Concurrent searches each open their own session from ``session_factory``:
    SQLAlchemy's AsyncSession is not safe for concurrent use, so the fan-out under
    asyncio.gather must never share one.
    """
    settings = get_agent_settings()

    query_shape = (
        chat_state.router_output.query_shape  # type: ignore[union-attr]
        if chat_state.router_output and hasattr(chat_state.router_output, "query_shape")
        else None
    )
    is_analytical = query_shape == "analytical"
    prompt_name, tools = prompt_and_tools(query_shape)
    max_iterations = settings.max_iterations_for(query_shape)
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
    # it was written to the chat tail before the task ran. Scan them here: strip invisibles/role markers as the current turn gets, and drop a blocked turn
    # outright rather than handing the agent the exact text the guardrail rejected.
    scan = get_injection_scan_user_input_enabled()
    history_messages = build_agent_history(history, scan=scan, request_id=request_id)
    messages: list[ChatMessage] = [
        ChatMessage(role=Role.system, content=system_content),
        # Uncapped, history is the largest unbounded cost in a turn: up to 50 prior
        # messages, resent every turn, dwarfing the excerpts they contextualize.
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
    # "searched and found nothing" from "never searched", on either path.
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
        settings=settings,
        max_iterations=max_iterations,
        transcript=Transcript(messages),
        expected_entities=expected_entities,
    )

    # One termination model for both paths. Extraction's plan is loop-authored — seeded
    # from the entities the router resolved, keyed by entity name so the model reports
    # under the name it was given. The analytical plan seeds itself from search
    # `sub_question`s instead.
    if not is_analytical:
        for name in sorted(expected_entities):
            state.plan[name] = name

    deps = RunDeps(
        llms=(llm, *fallbacks),
        chat_state=chat_state,
        session=session,
        session_factory=session_factory,
        reranker=reranker,
        redis_app=redis_app,
        request_id=request_id,
        is_analytical=is_analytical,
        search_sem=asyncio.Semaphore(settings.max_concurrent_searches),
        execute_search=execute_search,
        rewrite_model_id=rewrite_model_id,
    )
    # One wall-clock bound for the whole run. It cancels a turn mid-flight: that turn's
    # in-flight results are lost, while earlier turns are already in the ledgers.
    try:
        async with asyncio.timeout(settings.deadline_seconds):
            await _iterate(state, deps, tools)
    except TimeoutError:
        logger.warning(
            "agent_run_deadline",
            extra={
                "request_id": request_id,
                "iteration": state.iteration,
                "deadline_s": settings.deadline_seconds,
            },
        )
        state.convergence_reason = "deadline"
    iterations_run = state.iteration + 1

    logger.debug(
        "agent_synthesis_starting",
        extra={
            "request_id": request_id,
            "total_chunks": len(state.evidence),
            "iterations": iterations_run,
        },
    )

    meta = build_meta(
        state,
        iterations_run,
        prompt_version=prompt_name,
        rewrite_model=None if is_analytical else rewrite_model_id,
    )
    # Terminal state, attached to the enclosing "agent_loop" chain span (opened by the
    # caller, current here since no per-turn span is open past the loop) — otherwise a
    # non-converged/degraded run's final ledger contents are only reconstructable by
    # replaying every tool-call span in order.
    lf = lf_client.get_client()
    if lf:
        with contextlib.suppress(Exception):
            lf.update_current_span(metadata={"final_state": snapshot(state)})
    # Serve the ledger projection: a non-converged run yields its accumulated partial
    # findings marked degraded. None only when no report was ever attempted.
    # Keys still open are stated as limitations rather than vanishing. Only the analytical
    # envelope carries gaps; on extraction an unreported entity is surfaced by
    # `_inject_unsearched_stubs` instead.
    findings = state.findings.projection(
        analytical=is_analytical,
        degraded=not state.sealed,
        unresolved=unresolved_lines(state) if is_analytical else (),
    )
    return state.evidence, findings, meta
