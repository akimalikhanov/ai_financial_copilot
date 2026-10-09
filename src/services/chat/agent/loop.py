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
from typing import TYPE_CHECKING, Any
from uuid import UUID

from pydantic import ValidationError
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.observability import langfuse as lf_client
from src.observability.langfuse import describe_error
from src.observability.langfuse import mark as lf_mark
from src.observability.langfuse import mark_current as lf_mark_current
from src.observability.langfuse import span as lf_span
from src.observability.metrics import (
    AGENT_LAST_TURN_INPUT_TOKENS,
    AGENT_TOOL_ARG_ERRORS,
    AGENT_TOOL_CALLS,
    AGENT_TOOL_DURATION,
    AGENT_TOOL_MODEL_DURATION,
    AGENT_TOOL_MODEL_OUTPUT_TOKENS,
    LLM_CACHE_HIT_TOKENS,
    LLM_COST,
    LLM_TOKENS,
    observe_llm_failure,
    observe_llm_latency,
)
from src.observability.trace_payload import cap_list
from src.redis_client import add_event
from src.repository.llm_request_repository import LLMRequestRepository, stats_to_request_kwargs
from src.schemas.agent_findings import AgentFindings, Finding, FindingsReport
from src.schemas.retrieval import ChunkPromptPayload, RetrievedChunk
from src.services.chat.agent.evidence import EvidenceLedger
from src.services.chat.agent.state import (
    AgentLoopMeta,
    AgentRunState,
    AspectStats,
    ConvergenceReason,
    ShapeConfig,
    build_meta,
    get_agent_settings,
    open_aspects,
    render_status,
    shape_config,
    snapshot,
    unresolved_lines,
)
from src.services.chat.agent.tools import REPORT_TOOL_NAME, SearchDocumentsArgs
from src.services.chat.agent.transcript import Transcript
from src.services.chat.events import build_activity_event
from src.services.context.turns import as_messages, cap_turns, tool_model_history
from src.services.llm_adapters.base_adapter import (
    AssistantTurnResult,
    ChatMessage,
    Role,
    ToolCallRef,
)
from src.services.llm_runtime.exceptions import LLMError
from src.services.prompts.prompt_renderer import get_system_prompt
from src.services.retrieval.chat_rag import run_chat_rag_pipeline
from src.services.retrieval.payload_hydrator import get_chunk_prompt_payloads
from src.utils.config import get_agent_trace_chunk_chars

if TYPE_CHECKING:
    from src.schemas.chat import ChatPipelineState, Turn
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
    # Never conflate "the corpus was unreachable" with "the model sent bad
    # arguments" — only the former justifies Stop("search_unavailable") or a
    # couldn't-search gap. A malformed tool call is the model's problem, not the backend's.
    backend_failed: bool = False
    args_invalid: bool = False
    # The entity matched no document in the library, so no retrieval ran.
    not_found: bool = False
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
    [ToolCallRef, "ChatPipelineState", AsyncSession, "Reranker | None", Redis, str, int],
    Awaitable[_SearchResult],
]


@dataclass(frozen=True)
class RunDeps:
    """Everything a turn needs that is fixed for the whole run."""

    # The tool model first, then its fallbacks. A provider error moves a turn to the next.
    llms: tuple[RoutedLLM, ...]
    chat_state: ChatPipelineState
    # The pipeline's session. The loop does not write on it: sub-request rows and
    # concurrent searches each open their own session from `session_factory`.
    session: AsyncSession
    session_factory: async_sessionmaker[AsyncSession]
    reranker: Reranker | None
    redis_app: Redis
    request_id: str
    shape: ShapeConfig
    search_sem: asyncio.Semaphore
    execute_search: ExecuteSearchFn


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
    new_labels: int  # chunks shown to the model for the first time this turn
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


def _search_key(tc: ToolCallRef, state: AgentRunState, shape: ShapeConfig) -> str | None:
    """The plan key one search works on: an aspect id minted from its `sub_question`, or
    the seeded entity it names. None when it maps to no key.

    Deliberately tolerant: a call whose arguments don't parse still executes (and fails
    with its own error downstream), it just maps to no key.
    """
    try:
        args = SearchDocumentsArgs.model_validate_json(tc.arguments)
    except ValidationError:
        return None
    if shape.search_takes_sub_question:
        return _mint(state.plan, args.sub_question, state.settings.max_plan_items)
    return args.entity if args.entity in state.plan else None


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
        AGENT_TOOL_ARG_ERRORS.labels("search_documents").inc()
        return _SearchResult(
            entity="",
            chunks=[],
            payloads={},
            error_str="search_documents call had invalid arguments — entity and query are required strings.",
            args_invalid=True,
        )
    raw_query = search_args.query

    per_entity = (state.scope_result.per_entity_doc_ids or {}) if state.scope_result else {}
    entity = search_args.entity
    if entity in per_entity and not per_entity[entity]:
        # An entity the resolver could not match. Searching would mean searching nothing,
        # or worse, everything under its name — say so instead.
        AGENT_TOOL_CALLS.labels("search_documents", "error").inc()
        return _SearchResult(
            entity=entity,
            chunks=[],
            payloads={},
            error_str=f"{entity!r} did not match any document in your library.",
            not_found=True,
        )
    if entity and entity in per_entity:
        doc_ids = per_entity[entity]
    else:
        # entity="" (the analytical agent) searches every resolved entity's documents.
        # The SSE events and trace span carry the entity name only when there is one.
        doc_ids = state.scope_result.doc_ids if state.scope_result else None
        if not entity and len(per_entity) == 1:
            entity = next(iter(per_entity))

    _tool_started = perf_counter()
    activity_id, start_data = build_activity_event(
        "tool_call_started",
        label=entity,
        parent_id=f"round-{iteration}",
        detail={"tool": "search"},
    )
    await add_event(redis_app, request_id, "activity", start_data)

    # The tool model writes both retrievers' queries: `query` feeds the embedder and the
    # reranker, `keywords` feeds BM25. The schema requires `keywords`; a call that omits
    # it anyway still searches, with `query` on both legs.
    keyword_query = search_args.keywords or raw_query
    if not search_args.keywords:
        logger.warning("agent_search_keywords_missing", extra={"entity": entity})

    with lf_span(
        f"tool_search_{entity}_{iteration}",
        as_type="retriever",
        input={
            "entity": entity,
            "query": raw_query,
            "keywords": keyword_query,
            # Show the resolved scope this search was constrained to, so the
            # trace makes clear which docs the agent could actually see.
            "scope_doc_ids": cap_list([str(d) for d in doc_ids]) if doc_ids is not None else "all",
            "scope_doc_count": len(doc_ids) if doc_ids is not None else "all",
            "scoped_via_entity": entity in per_entity,
        },
    ) as obs:
        try:
            _, retrieval_trace, raw_chunks = await run_chat_rag_pipeline(
                session,
                semantic_query=raw_query,
                keyword_query=keyword_query,
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
                    lf_mark(obs, "ERROR", "all retrieval backends failed")
                AGENT_TOOL_CALLS.labels("search_documents", "error").inc()
                AGENT_TOOL_DURATION.labels("search_documents").observe(
                    perf_counter() - _tool_started
                )
                return _SearchResult(
                    entity=entity,
                    chunks=[],
                    payloads={},
                    error_str=f"Search failed for entity: {entity}",
                    backend_failed=True,
                    activity_id=activity_id,
                )
        except Exception as exc:
            logger.warning("agent_search_failed", extra={"entity": entity})
            if obs:
                obs.update(output={"chunks_returned": 0, "error": True})
                lf_mark(obs, "ERROR", describe_error(exc))
            AGENT_TOOL_CALLS.labels("search_documents", "error").inc()
            AGENT_TOOL_DURATION.labels("search_documents").observe(perf_counter() - _tool_started)
            return _SearchResult(
                entity=entity,
                chunks=[],
                payloads={},
                error_str=f"Search failed for entity: {entity}",
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
        degraded=degraded,
        scores_are_rerank=retrieval_trace.scores_are_rerank,
        activity_id=activity_id,
    )


# ---------------------------------------------------------------------------
# Report handling
# ---------------------------------------------------------------------------


def _resolve_refs(report: FindingsReport, state: AgentRunState, request_id: str) -> FindingsReport:
    """Rewrite each finding's `evidence` S-labels into chunk UUIDs.

    Unresolvable refs are dropped (never propagated downstream — a leaked label would
    surface in the synthesis prompt as a citable ID that has no matching excerpt).
    """
    all_unresolved: list[str] = []
    findings = []
    for f in report.findings:
        resolved, unresolved = state.evidence.resolve_refs(f.evidence)
        all_unresolved.extend(unresolved)
        findings.append(f.model_copy(update={"evidence": resolved}))
    if all_unresolved:
        logger.warning(
            "agent_chunk_refs_unresolved",
            extra={"request_id": request_id, "unresolved_refs": all_unresolved},
        )
    return report.model_copy(update={"findings": tuple(findings)})


def _searched(state: AgentRunState, key: str) -> bool:
    stats = state.aspect_stats.get(key)
    return stats is not None and stats.searches > 0


def _render_report_result(
    state: AgentRunState, closed: set[str], unknown: set[str], unsearched: set[str]
) -> str:
    """The tool result for one report: what landed and what didn't, nothing else.

    It never states coverage. The transcript is append-only, so an open list here would
    stay on screen and go stale; the status view (appended last on every call) is the
    only view of what is still open.
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
            f"{'their' if plural else 'its'} findings cited no chunk from the evidence "
            f"you retrieved. Re-report citing chunk labels from a search result, or, if the "
            f"documents do not support it, report it with supported: false."
        )
    if unknown:
        plural = len(unknown) != 1
        parts.append(
            f"{', '.join(repr(k) for k in sorted(unknown))} "
            f"{'are' if plural else 'is'} not an open key and "
            f"{'were' if plural else 'was'} not recorded. "
            "Use the key shown in brackets in a search result."
        )
    if not parts:
        parts.append("Nothing was recorded — no known key was reported.")
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
        parsed = FindingsReport.model_validate_json(tc.arguments)
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

    report = _resolve_refs(parsed, state, request_id)
    raw_keys = {f.key for f in report.findings}

    # Plan keys come only from the loop (seeded entities or minted aspect ids), so a key
    # it never created has no referent — it is named back rather than recorded.
    known = raw_keys & state.plan.keys()
    unknown = raw_keys - known
    state.unknown_aspect_keys += len(unknown)
    # A negative closes its key with nothing cited, so it must at least follow a search
    # for that key.
    unsearched = {
        f.key for f in report.findings if not f.supported and not _searched(state, f.key)
    } & known
    state.unsearched_negatives += len(unsearched)
    kept = tuple(
        f for f in report.findings if f.key in known and (f.supported or f.key not in unsearched)
    )

    # A reported-but-ungrounded key stays open: closing it would let `Stop("covered")`
    # seal a run that produced no grounded output for that key. Left open, it stays
    # searchable, and if it is still open at the end projection renders it as unresolved.
    state.findings.ingest(report.model_copy(update={"findings": kept}), state.evidence)

    AGENT_TOOL_CALLS.labels(tc.name, "ok").inc()
    return _render_report_result(state, closed=known, unknown=unknown, unsearched=unsearched)


# ---------------------------------------------------------------------------
# One turn
# ---------------------------------------------------------------------------


def fold_searches(
    state: AgentRunState,
    searches: list[ToolCallRef],
    results: list[_SearchResult],
    keys: dict[str, str | None],
) -> tuple[dict[str, str], list[int], dict[str, object]]:
    """Fold one turn's search results into the state, in call order.

    Returns the tool-result text per call id, the count of new labels each search
    showed the model, and a trace view per call id (chunk ids with trimmed text; the next
    turn's GENERATION input carries a longer, still capped, cut). No awaits and no I/O:
    labels are assigned here, sequentially, so S-labels continue across searches instead
    of restarting at S1, and stay deterministic however the concurrent searches finished.
    """
    texts: dict[str, str] = {}
    traces: dict[str, object] = {}
    new_per_search: list[int] = []
    trace_chars = get_agent_trace_chunk_chars()
    for tc, result in zip(searches, results, strict=True):
        entity_new = state.evidence.admit(result.chunks)
        state.degraded_capabilities |= result.degraded
        if result.chunks and not result.scores_are_rerank:
            state.scores_are_rerank = False
        if result.args_invalid:
            state.search_arg_errors += 1

        # Per-key search provenance, written at the one instant everything is in hand. A
        # failed or empty search admits no chunks, so this cannot be reconstructed from
        # the EvidenceLedger afterwards — and `unresolved_lines` needs it to tell "never
        # searched" and "backend down" from "not in the documents".
        aspect = keys.get(tc.id)
        if aspect is not None:
            stats = state.aspect_stats.setdefault(aspect, AspectStats())
            stats.searches += 1
            stats.new_chunks += entity_new
            if result.backend_failed:
                stats.errored += 1

        # A seeded entity with no documents can't produce evidence, so its search settles
        # the key as a stated negative; otherwise the key stays open until the budget runs out.
        if result.not_found and aspect is not None and aspect == result.entity:
            state.findings.record(
                aspect,
                Finding(
                    key=aspect,
                    claim=f"No document for {aspect} was found in the user's library.",
                    supported=False,
                    evidence=[],
                    confidence="high",
                ),
                state.evidence,
            )

        if result.error_str is not None:
            texts[tc.id] = traces[tc.id] = result.error_str
            new_per_search.append(0)
            continue
        # Admit the full result above for provenance, but render only the top-N into the
        # transcript — uncapped tool results are the biggest per-turn token cost. The
        # record stays complete; only the view is capped.
        top = result.chunks[: state.settings.max_chunks_per_entity]
        # A chunk an earlier search rendered is still on screen, so it is named, not
        # rendered again. Saying "no results" for it would send the model searching again.
        shown = state.evidence.shown_before(top)
        ctx = state.evidence.assign_labels(top, result.payloads)
        # Progress counts only what the model can read: a chunk admitted below the top-N
        # cut is never shown, so it can't make a near-repeat search look productive.
        new_per_search.append(len(ctx.items))
        parts = [ctx.formatted_context] if ctx.formatted_context else []
        if shown:
            parts.append("Already shown above: " + " · ".join(shown))
        body = "\n\n".join(parts) or "(no results)"
        # The key is echoed back so "reuse the key in brackets" is a copy from adjacent
        # context, not a slug reconstructed from memory.
        texts[tc.id] = f"[{aspect}] {body}" if aspect else body
        traces[tc.id] = {
            "aspect": aspect,
            "chunks": [
                {
                    "ref": item.ref_id,
                    "chunk_id": str(item.chunk_id),
                    "text": item.prompt_text[:trace_chars],
                }
                for item in ctx.items
            ],
            "already_shown": shown,
        }
    return texts, new_per_search, traces


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
    # new_labels == 0 and the model is told to reformulate while burning its budget.
    if facts.searches and facts.backend_failures == facts.searches:
        return Stop("search_unavailable")

    # Progress = a new label shown *or* a key closed. A turn that settles a key from
    # evidence already in hand is real progress; without this, resolving the last aspects
    # from earlier evidence trips convergence one turn before coverage.
    if facts.new_labels == 0 and not facts.closed:
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


def _call_budget(state: AgentRunState) -> float:
    """Seconds the next tool-model call may take: what is left of the run deadline less
    the reserve, capped. The deadline belongs to the request, so each call inherits what
    remains of it instead of a fixed timeout picked when calls were shorter."""
    settings = state.settings
    if state.deadline_at is None:
        return settings.turn_timeout_cap_seconds
    remaining = state.deadline_at - asyncio.get_running_loop().time()
    return min(settings.turn_timeout_cap_seconds, remaining - settings.deadline_reserve_seconds)


async def _call_tool_model(
    state: AgentRunState,
    deps: RunDeps,
    messages: list[ChatMessage],
    tools: list[dict],
    allowed: list[str] | None = None,
) -> tuple[RoutedLLM, AssistantTurnResult]:
    """The first model in the chain that answers, and its turn. The last model's provider
    error propagates; a timeout does not fall back. `allowed` restricts the callable tools:
    via `allowed_tools` where the model supports it (keeps the cached prefix), otherwise by
    dropping the other tools from the request."""

    async def complete(llm: RoutedLLM) -> AssistantTurnResult:
        if allowed is None:
            call = llm.complete_with_tools(messages, tools=tools, temperature=0.0)
        elif llm.capabilities.get("allowed_tools", False):
            call = llm.complete_with_tools(
                messages, tools=tools, allowed_tools=allowed, temperature=0.0
            )
        else:
            subset = [t for t in tools if t["function"]["name"] in allowed]
            call = llm.complete_with_tools(messages, tools=subset, temperature=0.0)
        # Recomputed per call: a fallback gets only what the failed attempt left.
        budget = _call_budget(state)
        started = perf_counter()
        # A call that never answers has no stats, so without these it leaves no row and no
        # latency sample: the slowest calls would be missing from every percentile.
        # "timeout" is this call's own budget; "cancelled" is a cancel from outside (a
        # worker shutdown — the reserve keeps the run deadline from landing mid-call).
        try:
            return await asyncio.wait_for(call, timeout=max(budget, 0.0))
        except TimeoutError:
            await _finish_despite_cancel(
                _record_failed_call(state, deps, llm, allowed, "timeout", started, budget)
            )
            raise
        except asyncio.CancelledError:
            await _finish_despite_cancel(
                _record_failed_call(state, deps, llm, allowed, "cancelled", started, budget)
            )
            raise
        except LLMError as e:
            await _finish_despite_cancel(
                _record_failed_call(state, deps, llm, allowed, "error", started, budget, e)
            )
            raise

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
            lf_mark_current("WARNING", f"tool model {llm.model_id} failed ({type(e).__name__})")
    return last, await complete(last)


async def _finish_despite_cancel(aw: Awaitable[None]) -> None:
    """Run `aw` to completion even if the run deadline cancels this task meanwhile, then
    let the cancellation through. A row write interrupted mid-commit would lose the row."""
    task = asyncio.ensure_future(aw)
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise


def _turn_kind(allowed: list[str] | None, turn: AssistantTurnResult | None) -> str:
    """What a tool-model call did, for splitting its latency. `turn` is None when the call
    failed before answering: only the final turn's kind is known in advance."""
    if allowed is not None:
        return "final"
    if turn is None:
        return "unknown"
    names = {tc.name for tc in turn.tool_calls or []}
    if not names:
        return "none"
    if REPORT_TOOL_NAME not in names:
        return "search"
    return "report" if names == {REPORT_TOOL_NAME} else "mixed"


async def _write_tool_call_row(
    state: AgentRunState,
    deps: RunDeps,
    llm: RoutedLLM,
    request_params: dict,
    **row: Any,
) -> None:
    """The `llm_requests` sub-request row for one tool-model call. Best effort: a failed
    write never fails the turn.

    Written on its own short-lived session, committed at once, so the connection is back
    in the pool before the next LLM call. A failure stays in that session: rolling back
    the shared `deps.session` instead would expire every object the pipeline holds
    (`llm_request` included — a later attribute read then raises MissingGreenlet) and
    discard its pending changes."""
    llm_request = deps.chat_state.llm_request
    if llm_request is None or llm_request.conversation_id is None:
        return
    try:
        async with deps.session_factory() as session:
            await LLMRequestRepository(session).create_subrequest(
                parent_request_id=llm_request.id,
                conversation_id=llm_request.conversation_id,
                user_id=llm_request.user_id,
                provider=llm.provider,
                model=llm.model_id,
                request_type="agent_tool_call",
                request_params={"iteration": state.iteration, **request_params},
                **row,
            )
            await session.commit()
    except Exception:
        logger.warning(
            "agent_tool_call_row_failed",
            extra={"request_id": deps.request_id, "iteration": state.iteration},
            exc_info=True,
        )


async def _record_failed_call(
    state: AgentRunState,
    deps: RunDeps,
    llm: RoutedLLM,
    allowed: list[str] | None,
    outcome: str,
    started: float,
    budget: float,
    error: LLMError | None = None,
) -> None:
    """Metrics and the `llm_requests` row for a tool-model call that never answered.
    The tokens a timed-out call was billed for are unknown, so the row has none."""
    elapsed = perf_counter() - started
    turn_kind = _turn_kind(allowed, None)
    observe_llm_failure(llm.model_id, "agent_tool_call", outcome, elapsed)
    AGENT_TOOL_MODEL_DURATION.labels(llm.model_id, turn_kind, outcome).observe(elapsed)
    if outcome == "timeout":
        logger.warning(
            "agent_tool_model_timeout",
            extra={
                "request_id": deps.request_id,
                "iteration": state.iteration,
                "model": llm.model_id,
                "budget_s": round(budget, 1),
                # Below the cap means the run deadline, not the cap, set this budget.
                "cap_s": state.settings.turn_timeout_cap_seconds,
            },
        )
    await _write_tool_call_row(
        state,
        deps,
        llm,
        {"turn_kind": turn_kind, "budget_s": round(budget, 1)},
        status="failed" if outcome == "error" else outcome,
        latency_ms=int(elapsed * 1000),
        error_code=type(error).__name__ if error else None,
        error_message=describe_error(error) if error else None,
    )


async def _record_turn_spend(
    state: AgentRunState,
    deps: RunDeps,
    llm: RoutedLLM,
    turn: AssistantTurnResult,
    allowed: list[str] | None,
) -> None:
    """Spend, metrics and the `llm_requests` sub-request row for one tool-model call."""
    stats = turn.stats
    if stats is None:
        return
    turn_kind = _turn_kind(allowed, turn)
    state.record_spend(llm.model_id, stats)
    state.last_turn_input_tokens = stats.input_tokens or 0
    if stats.input_tokens:
        LLM_TOKENS.labels("input", llm.model_id).inc(stats.input_tokens)
    if stats.output_tokens:
        LLM_TOKENS.labels("output", llm.model_id).inc(stats.output_tokens)
    if stats.cached_input_tokens:
        LLM_CACHE_HIT_TOKENS.labels(llm.model_id).inc(stats.cached_input_tokens)
    if stats.cost_usd:
        LLM_COST.labels(llm.model_id).inc(stats.cost_usd)
    # Hitting the completion-token cap is its own failure: reasoning can spend the whole
    # cap before any tool call is written, and the turn then looks like a prose answer.
    truncated = turn.finish_reason == "length"
    outcome = "length" if truncated else "ok"
    if truncated:
        logger.warning(
            "agent_tool_model_truncated",
            extra={
                "request_id": deps.request_id,
                "iteration": state.iteration,
                "model": llm.model_id,
                "output_tokens": stats.output_tokens,
                "reasoning_tokens": stats.reasoning_tokens,
                "tool_calls": len(turn.tool_calls or []),
            },
        )
        lf_mark_current("WARNING", f"tool model hit its token cap ({stats.output_tokens} out)")
    observe_llm_latency(llm.model_id, "agent_tool_call", stats, outcome)
    if stats.latency_ms is not None:
        AGENT_TOOL_MODEL_DURATION.labels(llm.model_id, turn_kind, outcome).observe(
            stats.latency_ms / 1000.0
        )
    if stats.output_tokens:
        reasoning = stats.reasoning_tokens or 0
        AGENT_TOOL_MODEL_OUTPUT_TOKENS.labels(llm.model_id, turn_kind, "reasoning").inc(reasoning)
        AGENT_TOOL_MODEL_OUTPUT_TOKENS.labels(llm.model_id, turn_kind, "visible").inc(
            max(stats.output_tokens - reasoning, 0)
        )

    await _write_tool_call_row(
        state,
        deps,
        llm,
        {"tool_calls_issued": len(turn.tool_calls or []), "turn_kind": turn_kind},
        status="truncated" if truncated else "completed",
        **stats_to_request_kwargs(stats),
    )


async def _guarded_search(tc: ToolCallRef, state: AgentRunState, deps: RunDeps) -> _SearchResult:
    """One search, bounded by the search timeout; a timeout becomes a backend failure."""
    try:
        async with asyncio.timeout(state.settings.search_timeout_seconds):
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
                )
    except TimeoutError:
        logger.warning(
            "agent_search_timeout",
            extra={
                "request_id": deps.request_id,
                "iteration": state.iteration,
                "timeout_s": state.settings.search_timeout_seconds,
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


async def _run_turn(
    state: AgentRunState, deps: RunDeps, tools: list[dict], allowed: list[str] | None = None
) -> TurnOutcome:
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
    new_labels = 0
    turn: AssistantTurnResult | None = None
    results_by_id: dict[str, str] = {}
    search_traces: dict[str, object] = {}
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
            served, turn = await _call_tool_model(state, deps, prompt_messages, tools, allowed)
            await _finish_despite_cancel(_record_turn_spend(state, deps, served, turn, allowed))

            if not turn.tool_calls:
                # With no terminal tool this is a normal exit, not a rare one: the model
                # emitted prose instead of a call. projection() still serves whatever the
                # ledger accumulated (None only if nothing was ever reported). A turn cut
                # off by the token cap looks the same and is told apart here.
                return Stop("truncated" if turn.finish_reason == "length" else "natural")

            # A call to a tool outside this turn's pool is answered and never parsed: a
            # final-turn search does not run.
            offered = set(allowed) if allowed else {t["function"]["name"] for t in tools}
            reports = [tc for tc in turn.tool_calls if tc.name == REPORT_TOOL_NAME]
            searches = [tc for tc in turn.tool_calls if tc.name in offered - {REPORT_TOOL_NAME}]
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

            # Key before execution, so the tool result can echo the key the model must cite.
            keys = {tc.id: _search_key(tc, state, deps.shape) for tc in searches}

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
            before_searches = state.addressed
            search_texts, new_per_search, search_traces = fold_searches(
                state, searches, results, keys
            )
            closed |= state.addressed - before_searches
            results_by_id |= search_texts
            new_labels = sum(new_per_search)
            for tc, result, search_new in zip(searches, results, new_per_search, strict=True):
                logger.debug(
                    "tool_call_completed",
                    extra={
                        "request_id": request_id,
                        "iteration": iteration,
                        "entity": result.entity,
                        "aspect": keys.get(tc.id),
                        "chunks_returned": len(result.chunks),
                        "new_labels": search_new,
                    },
                )
                if result.activity_id is not None:
                    _, end_data = build_activity_event(
                        "tool_call_ended",
                        event_id=result.activity_id,
                        detail={
                            "chunks_returned": len(result.chunks),
                            "new_chunks_added": search_new,
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

            return decide(
                state,
                TurnFacts(
                    searches=len(searches),
                    backend_failures=sum(r.backend_failed for r in results),
                    new_labels=new_labels,
                    closed=closed,
                ),
            )
        finally:
            if obs:
                obs.update(
                    output={
                        "tool_calls": [
                            {"name": tc.name, "arguments": tc.arguments}
                            for tc in (turn.tool_calls if turn and turn.tool_calls else [])
                        ],
                        # The `role: tool` results this turn appended, keyed by
                        # tool_call_id. Report results verbatim; a search as chunk ids with
                        # trimmed text — the next turn's GENERATION input logs it once.
                        "tool_results": {**results_by_id, **search_traces},
                        # State *after* this turn's effects landed — the structured
                        # counterpart to the prose `status` this span took as input (state
                        # *before* the turn ran).
                        "state_after": snapshot(state),
                    },
                    metadata={
                        "token_spend_cumulative": state.input_tokens_total(),
                        "cost_usd_cumulative": state.cost_usd_total(),
                        "new_labels": new_labels,
                    },
                )


CARRYOVER_STUB = "[prior turn: restated earlier results in a different format]"


def agent_history(turns: Sequence[Turn]) -> list[ChatMessage]:
    """Prior turns as the tool model sees them, capped to `tool_model_history()`.

    An answer derived from a carried block is stubbed, since its prose restates numbers
    the agent has no evidence for (Contract F1). Stubbing happens before capping, so the
    budget counts what is actually sent.
    """
    stubbed = [
        t.model_copy(update={"answer": CARRYOVER_STUB})
        if t.from_carryover and t.answer is not None
        else t
        for t in turns
    ]
    return as_messages(cap_turns(stubbed, tool_model_history()))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


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
        # already holds, at no extra cost. The tool mode stays "auto": if the model emits
        # prose instead, the turn returns Stop("natural") and accumulated findings still
        # serve, so this fails safe.
        final_turn = iteration == state.max_iterations - 1
        allowed = [REPORT_TOOL_NAME] if final_turn else None
        # A call with no budget left would only be cut off by the deadline mid-flight.
        if _call_budget(state) <= 0:
            logger.warning(
                "agent_run_out_of_time",
                extra={"request_id": request_id, "iteration": iteration},
            )
            state.convergence_reason = "deadline"
            return
        try:
            outcome = await _run_turn(state, deps, tools, allowed)
        except TimeoutError:
            # The call's budget and the cap are on the agent_tool_model_timeout line.
            logger.warning(
                "agent_turn_timeout", extra={"request_id": request_id, "iteration": iteration}
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
    AgentFindings | None,
    AgentLoopMeta,
]:
    """Run the agent tool-calling loop for retrieval queries.

    Returns (evidence, agent_findings, meta). The ledger carries the ordered chunks, the
    payloads cached at render time (so synthesis hydrates nothing), and the rendered
    subset — what the model could actually read, the only defensible pool for synthesis to
    fall back on. agent_findings is the FindingsLedger projection — the accumulated
    findings, marked degraded when the run did not seal, and None only when no report
    was ever attempted and no plan key is open.

    ``fallbacks`` are tried in order when ``llm`` raises a provider error. When the whole
    chain fails, the run stops as ``llm_error`` and serves what it gathered; with nothing
    gathered the error propagates and the request fails.

    The loop never writes on the caller's ``session``. Each sub-request row and each
    concurrent search opens its own session from ``session_factory``: a failed row write
    then cannot abort the pipeline's transaction, and SQLAlchemy's AsyncSession is not
    safe for concurrent use, so the fan-out under asyncio.gather must never share one.
    """
    settings = get_agent_settings()

    query_shape = (
        chat_state.router_output.query_shape  # type: ignore[union-attr]
        if chat_state.router_output and hasattr(chat_state.router_output, "query_shape")
        else None
    )
    shape = shape_config(query_shape, settings)

    system_content = get_system_prompt(version=shape.prompt)
    messages: list[ChatMessage] = [
        ChatMessage(role=Role.system, content=system_content),
        *agent_history(chat_state.prior_turns),
    ]

    # Inject exact entity names from the router so the agent uses correct strings and
    # knows which entities it must cover before calling report_findings. Also inject
    # metadata-backed years so the agent does not hallucinate fiscal year terms.
    _entity_years: dict[str, list[int]] = {}
    mentions = chat_state.scope_result.mentions() if chat_state.scope_result else {}
    if chat_state.scope_result and chat_state.scope_result.entity_manifest:
        for item in chat_state.scope_result.entity_manifest:
            years = sorted(
                {s["year"] for s in (item.doc_summaries or []) if s.get("year")},
                reverse=True,
            )
            if years:
                _entity_years[item.entity_name] = years

    expected_entities: set[str] = set()
    not_found: set[str] = set()
    if chat_state.scope_result and chat_state.scope_result.per_entity_doc_ids:
        per_entity = chat_state.scope_result.per_entity_doc_ids
        expected_entities = set(per_entity)
        not_found = {name for name, ids in per_entity.items() if not ids}

    # The task is one user message: the scope (entities and years) above the question.
    scope_block: str | None = None
    if shape.seed_plan_from_entities and expected_entities:
        lines: list[str] = []
        for name in sorted(expected_entities):
            years = _entity_years.get(name)
            if name in not_found:
                suffix = " (not found in your documents)"
            elif years:
                suffix = f" (available years: {', '.join(str(y) for y in years)})"
            else:
                suffix = ""
            if name in mentions:
                suffix = f" (the question calls it {mentions[name]}){suffix}"
            lines.append(f"- {name}{suffix}")
        scope_block = (
            "Entities to search (you MUST call search_documents for each before report_findings).\n"
            "Use ONLY the listed years in your search queries — do not guess or invent fiscal years:\n"
            + "\n".join(lines)
        )
    elif not shape.seed_plan_from_entities and _entity_years:
        year_lines = [
            f"- {name}: {', '.join(str(y) for y in years)}"
            for name, years in sorted(_entity_years.items())
        ]
        scope_block = (
            "Available document years (use ONLY these in search queries — do not invent fiscal years):\n"
            + "\n".join(year_lines)
        )
    if not shape.seed_plan_from_entities and not_found:
        scope_block = (f"{scope_block}\n" if scope_block else "") + (
            f"Not found in your documents: {', '.join(sorted(not_found))}"
        )

    task = chat_state.user_query_raw
    if scope_block is not None:
        task = f"{scope_block}\n\nQuestion: {chat_state.user_query_raw}"
    messages.append(ChatMessage(role=Role.user, content=task))

    state = AgentRunState(
        settings=settings,
        max_iterations=shape.max_iterations,
        transcript=Transcript(messages),
    )

    # One termination model for both shapes. A seeded plan is keyed by the entity names
    # the router resolved, so the model reports under the name it was given; otherwise the
    # plan grows from search `sub_question`s. An entity never searched is then an open key
    # like any other, and `unresolved_lines` says so.
    if shape.seed_plan_from_entities:
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
        shape=shape,
        search_sem=asyncio.Semaphore(settings.max_concurrent_searches),
        execute_search=execute_search,
    )
    # One wall-clock bound for the whole run. Tool-model calls are budgeted to end before
    # it; a search still running when it fires is cancelled, and its results are lost,
    # while earlier turns are already in the ledgers.
    try:
        async with asyncio.timeout(settings.deadline_seconds) as deadline:
            state.deadline_at = deadline.when()
            await _iterate(state, deps, shape.tools)
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
    if state.last_turn_input_tokens:
        AGENT_LAST_TURN_INPUT_TOKENS.observe(state.last_turn_input_tokens)

    logger.debug(
        "agent_synthesis_starting",
        extra={
            "request_id": request_id,
            "total_chunks": len(state.evidence),
            "iterations": iterations_run,
        },
    )

    meta = build_meta(state, iterations_run, prompt_version=shape.prompt)
    # Terminal state, attached to the enclosing "agent_loop" chain span (opened by the
    # caller, current here since no per-turn span is open past the loop) — otherwise a
    # non-converged/degraded run's final ledger contents are only reconstructable by
    # replaying every tool-call span in order.
    lf = lf_client.get_client()
    if lf:
        with contextlib.suppress(Exception):
            lf.update_current_span(metadata={"final_state": snapshot(state)})
    # Serve the ledger projection: a non-converged run yields its accumulated partial
    # findings marked degraded, and keys still open are stated as limitations rather than
    # vanishing.
    findings = state.findings.projection(
        degraded=not state.sealed, unresolved=unresolved_lines(state)
    )
    return state.evidence, findings, meta
