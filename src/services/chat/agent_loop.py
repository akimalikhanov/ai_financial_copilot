"""Agent loop for agentic RAG: drives search_documents tool calls, collects chunks."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from dataclasses import dataclass, replace
from time import perf_counter
from typing import TYPE_CHECKING, Literal
from uuid import UUID

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

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
from src.schemas.agent_findings import AgentFindings, AnalyticalFindings, EntityFinding, Observation
from src.schemas.query_transform import ScopeDocSummary, TransformedQuery
from src.schemas.retrieval import ChunkPromptPayload, RetrievedChunk
from src.services.llm_adapters.base_adapter import (
    AssistantTurnResult,
    ChatMessage,
    LLMResponseStats,
    Role,
    ToolCallRef,
)
from src.services.prompts.prompt_renderer import get_system_prompt
from src.services.retrieval.chat_rag import run_chat_rag_pipeline
from src.services.retrieval.context_assembler import assemble_rag_context
from src.services.retrieval.payload_hydrator import get_chunk_prompt_payloads
from src.services.retrieval.query_transformer import rewrite_query
from src.utils.config import get_agent_config

if TYPE_CHECKING:
    from src.schemas.chat import ChatPipelineState
    from src.services.llm_router import RoutedLLM
    from src.services.retrieval.reranker import Reranker

logger = logging.getLogger(__name__)

_AGENT_TURN_TIMEOUT = float(os.getenv("AGENT_TURN_TIMEOUT_SECONDS", "60"))

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

_SEARCH_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": "Search financial documents for a specific entity. Call once per entity.",
        "parameters": {
            "type": "object",
            "required": ["entity", "query"],
            "properties": {
                "entity": {"type": "string"},
                "query": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
}

_REPORT_FINDINGS_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "report_findings",
        "description": "Call this once when you have finished searching. Report extracted values for all entities. This ends the search phase.",
        "parameters": {
            "type": "object",
            "required": ["metric_requested", "findings"],
            "properties": {
                "metric_requested": {"type": "string"},
                "comparison_op": {
                    "type": ["string", "null"],
                    "enum": ["argmin", "argmax", "list", "none", None],
                },
                "findings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["entity", "available"],
                        "properties": {
                            "entity": {"type": "string"},
                            "value": {"type": ["number", "null"]},
                            "currency": {"type": ["string", "null"]},
                            "period_end": {"type": ["string", "null"]},
                            "source_chunks": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": 'Excerpt IDs from search results that contain the value, exactly as shown (e.g. ["S3", "S7"]).',
                            },
                            "available": {"type": "boolean"},
                            "reason": {"type": ["string", "null"]},
                            "unit": {
                                "type": ["string", "null"],
                                "description": "Scale suffix as stated in the document: 'M' for millions, 'B' for billions, 'K' for thousands, '' for absolute values.",
                            },
                        },
                        "additionalProperties": False,
                    },
                },
            },
            "additionalProperties": False,
        },
    },
}

_REPORT_ANALYTICAL_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "report_analytical_findings",
        "description": "Call this once when you have a complete chain of observations for a causal or narrative question. This ends the search phase.",
        "parameters": {
            "type": "object",
            "required": ["question", "observations"],
            "properties": {
                "question": {"type": "string"},
                "conclusion": {"type": ["string", "null"]},
                "gaps": {"type": ["array", "null"], "items": {"type": "string"}},
                "observations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["claim", "evidence_chunks", "confidence"],
                        "properties": {
                            "claim": {"type": "string"},
                            "evidence_chunks": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": 'Excerpt IDs from search results that support this claim, exactly as shown (e.g. ["S3", "S7"]).',
                            },
                            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                            "refuted_by": {
                                "type": ["array", "null"],
                                "items": {"type": "string"},
                                "description": "Excerpt IDs that contradict this claim, exactly as shown in search results.",
                            },
                        },
                        "additionalProperties": False,
                    },
                },
            },
            "additionalProperties": False,
        },
    },
}

_CONVERT_CURRENCY_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "convert_currency",
        "description": (
            "Convert a numeric amount between currencies at a given date. "
            "Use ONLY when you already have the value from context and just need unit conversion. "
            "Do NOT use this to retrieve financial data — use search_documents for that."
        ),
        "parameters": {
            "type": "object",
            "required": ["amount", "from_currency", "to_currency", "date"],
            "properties": {
                "amount": {"type": "number"},
                "from_currency": {"type": "string"},
                "to_currency": {"type": "string"},
                "date": {"type": "string", "description": "ISO 8601 date (YYYY-MM-DD) or 'latest'"},
            },
            "additionalProperties": False,
        },
    },
}

# All FX runs through router-extracted requested_currency → deterministic trigger in
# findings_processor. _CONVERT_CURRENCY_TOOL / _execute_convert_currency are intentionally
# kept but NOT exposed to the agent: add _CONVERT_CURRENCY_TOOL to the list below to revive.
TOOLS_EXTRACTION_COMPARISON = [_SEARCH_TOOL, _REPORT_FINDINGS_TOOL]
TOOLS_ANALYTICAL = [_SEARCH_TOOL, _REPORT_ANALYTICAL_TOOL]

_FINALIZER_NAMES = frozenset({"report_findings", "report_analytical_findings"})


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class AgentLoopMeta:
    iterations: int
    tool_calls_total: int
    convergence_reason: Literal["natural", "convergence", "iteration_cap", "budget_cap", "timeout"]
    currency_normalized: bool = False
    input_tokens_total: int = 0
    output_tokens_total: int = 0
    cost_usd_total: float = 0.0


@dataclass
class _SearchResult:
    chunks: list[RetrievedChunk]
    # Hydrated payloads for chunks — context is assembled later, sequentially, so
    # S-labels can be numbered globally across all searches in the request.
    payloads: dict[UUID, ChunkPromptPayload]
    error_str: str | None = None
    rewrite_stats: LLMResponseStats | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _order_chunks(registry: dict[UUID, RetrievedChunk]) -> list[RetrievedChunk]:
    return sorted(registry.values(), key=lambda c: (c.turn_index, -(c.score or 0)))


def _compress_history(messages: list[ChatMessage], keep_last_n_turns: int = 2) -> list[ChatMessage]:
    """Replace tool results from turns older than keep_last_n_turns with compact stubs.

    A "turn" is an assistant message that contains tool_calls followed by its tool
    result messages. We truncate whole turns so the agent never sees a partial view
    of a prior turn's evidence.
    """
    # Locate turn boundaries: indices of assistant messages that issued tool calls
    turn_starts: list[int] = [
        i for i, m in enumerate(messages) if m.role == Role.assistant and m.tool_calls
    ]
    if len(turn_starts) <= keep_last_n_turns:
        return messages

    # Everything before the (N-th from last) turn start is stale
    cutoff_idx = turn_starts[-(keep_last_n_turns)]
    stale: set[int] = set()
    for i, m in enumerate(messages):
        if i >= cutoff_idx:
            break
        if m.role in (Role.tool, Role.assistant) and (m.role == Role.tool or m.tool_calls):
            stale.add(i)

    result = []
    for i, m in enumerate(messages):
        if i in stale and m.role == Role.tool:
            result.append(
                ChatMessage(
                    role=m.role,
                    tool_call_id=m.tool_call_id,
                    content="[turn results truncated — already incorporated]",
                )
            )
        elif i in stale and m.role == Role.assistant:
            # Keep the assistant message so tool_call_id references stay valid
            result.append(m)
        else:
            result.append(m)
    return result


def _assistant_msg_with_tool_calls(tool_calls: list[ToolCallRef]) -> ChatMessage:
    return ChatMessage(
        role=Role.assistant,
        content=None,
        tool_calls=tuple(tool_calls),
    )


async def _execute_search(
    tc: ToolCallRef,
    state: ChatPipelineState,
    session: AsyncSession,
    reranker: Reranker | None,
    redis_app: Redis,
    request_id: str,
    iteration: int,
) -> _SearchResult:
    args = json.loads(tc.arguments)
    entity: str = args.get("entity", "unknown")
    raw_query: str = args.get("query", "")

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
            chunks=[],
            payloads={},
            error_str=f"Search failed for entity: {entity}",
            rewrite_stats=rewrite_stats,
        )
    finally:
        _search_lf_stack.close()

    chunks = [replace(c, turn_index=iteration) for c in raw_chunks]
    payloads = await get_chunk_prompt_payloads(session, [c.chunk_id for c in chunks])
    AGENT_TOOL_CALLS.labels("search_documents", "ok").inc()
    AGENT_TOOL_DURATION.labels("search_documents").observe(perf_counter() - _tool_started)
    return _SearchResult(chunks=chunks, payloads=payloads, rewrite_stats=rewrite_stats)


async def _execute_convert_currency(
    amount: float,
    from_currency: str,
    to_currency: str,
    date: str,
) -> str:
    import httpx

    from src.services.chat.findings_processor import _FRANKFURTER_BASE, _FX_TIMEOUT, _normalize_date

    if from_currency == to_currency:
        return f"{amount:,.4f} {to_currency} (same currency, no conversion needed)"

    normalized_date = _normalize_date(date) if date != "latest" else None
    date_str = normalized_date or "latest"
    key = f"{from_currency}->{to_currency}@{date_str}"
    try:
        async with httpx.AsyncClient(timeout=_FX_TIMEOUT) as client:
            r = await client.get(
                f"{_FRANKFURTER_BASE}/{date_str}",
                params={"from": from_currency, "to": to_currency},
            )
            r.raise_for_status()
            rate = r.json()["rates"].get(to_currency)
        if rate is None:
            return f"FX rate not available for {key}"
        converted = amount * rate
        return f"{amount:,.4f} {from_currency} = {converted:,.4f} {to_currency} (rate: {rate:.6f}, date: {date_str})"
    except Exception as exc:
        logger.warning("convert_currency_failed %s: %s", key, exc)
        return f"FX conversion failed for {key}: {exc}"


def _parse_findings(tc: ToolCallRef) -> AgentFindings | AnalyticalFindings:
    data = json.loads(tc.arguments)
    if tc.name == "report_findings":
        raw_findings = data.get("findings", [])
        findings = tuple(
            EntityFinding(
                entity=f["entity"],
                available=f["available"],
                value=f.get("value"),
                currency=f.get("currency"),
                period_end=f.get("period_end"),
                source_chunks=f.get("source_chunks") or [],
                reason=f.get("reason"),
                unit=f.get("unit"),
            )
            for f in raw_findings
        )
        return AgentFindings(
            metric_requested=data.get("metric_requested", ""),
            findings=findings,
            comparison_op=data.get("comparison_op"),
        )
    # report_analytical_findings
    observations = tuple(
        Observation(
            claim=o["claim"],
            evidence_chunks=o.get("evidence_chunks", []),
            confidence=o["confidence"],
            refuted_by=o.get("refuted_by"),
        )
        for o in data.get("observations", [])
    )
    return AnalyticalFindings(
        question=data.get("question", ""),
        observations=observations,
        conclusion=data.get("conclusion"),
        gaps=data.get("gaps"),
    )


def _resolve_chunk_refs(
    refs: list[str] | None,
    ref_registry: dict[str, UUID],
) -> tuple[list[str], list[str]]:
    """Map agent-reported chunk refs to chunk UUID strings.

    The agent only ever sees excerpts labeled S1..Sn in tool results, so it reports
    those labels (not UUIDs). Resolve labels via the per-request registry; pass
    through anything that is already a UUID. Returns (resolved, unresolved).
    """
    resolved: list[str] = []
    unresolved: list[str] = []
    for ref in refs or []:
        candidate = ref.strip()
        try:
            UUID(candidate)
        except ValueError:
            chunk_id = ref_registry.get(candidate.upper())
            if chunk_id is not None:
                resolved.append(str(chunk_id))
            else:
                unresolved.append(candidate)
        else:
            resolved.append(candidate)
    return resolved, unresolved


def _resolve_finding_refs(
    findings: AgentFindings | AnalyticalFindings,
    ref_registry: dict[str, UUID],
    request_id: str,
) -> AgentFindings | AnalyticalFindings:
    """Rewrite source_chunks / evidence_chunks / refuted_by S-labels into chunk UUIDs.

    Unresolvable refs are dropped (never propagated downstream — a leaked label would
    surface in the synthesis prompt as a citable ID that has no matching excerpt).
    """
    all_unresolved: list[str] = []
    result: AgentFindings | AnalyticalFindings

    if isinstance(findings, AgentFindings):
        new_findings: list[EntityFinding] = []
        for f in findings.findings:
            resolved, unresolved = _resolve_chunk_refs(f.source_chunks, ref_registry)
            all_unresolved.extend(unresolved)
            new_findings.append(replace(f, source_chunks=resolved))
        result = replace(findings, findings=tuple(new_findings))
    else:
        new_obs: list[Observation] = []
        for o in findings.observations:
            evidence, unresolved = _resolve_chunk_refs(o.evidence_chunks, ref_registry)
            all_unresolved.extend(unresolved)
            refuted: list[str] | None = o.refuted_by
            if o.refuted_by is not None:
                refuted, unresolved = _resolve_chunk_refs(o.refuted_by, ref_registry)
                all_unresolved.extend(unresolved)
            new_obs.append(replace(o, evidence_chunks=evidence, refuted_by=refuted))
        result = replace(findings, observations=tuple(new_obs))

    if all_unresolved:
        logger.warning(
            "agent_chunk_refs_unresolved",
            extra={"request_id": request_id, "unresolved_refs": all_unresolved},
        )
    return result


def _finding_chunk_ids(findings: AgentFindings | AnalyticalFindings) -> set[str]:
    """All chunk-id strings referenced by the findings."""
    ids: set[str] = set()
    if isinstance(findings, AgentFindings):
        for f in findings.findings:
            ids.update(f.source_chunks or [])
    else:
        for o in findings.observations:
            ids.update(o.evidence_chunks or [])
            ids.update(o.refuted_by or [])
    return ids


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def run_agent_loop(
    state: ChatPipelineState,
    llm: RoutedLLM,
    session: AsyncSession,
    redis_app: Redis,
    request_id: str,
    reranker: Reranker | None,
) -> tuple[dict[UUID, RetrievedChunk], AgentFindings | AnalyticalFindings | None, AgentLoopMeta]:
    """Run the agent tool-calling loop for retrieval queries.

    Returns (chunk_registry, agent_findings, meta).
    chunk_registry is keyed by chunk_id; values have turn_index stamped.
    agent_findings is None when the agent hit an iteration/budget cap.
    """
    cfg = get_agent_config()
    max_iterations: int = cfg["max_iterations"]
    token_budget: int = cfg["token_budget"]
    max_chunks_per_entity: int = cfg["max_chunks_per_entity"]
    _search_sem = asyncio.Semaphore(cfg["max_concurrent_searches"])

    query_shape = (
        state.router_output.query_shape  # type: ignore[union-attr]
        if state.router_output and hasattr(state.router_output, "query_shape")
        else None
    )
    is_analytical = query_shape == "analytical"
    tools = TOOLS_ANALYTICAL if is_analytical else TOOLS_EXTRACTION_COMPARISON
    prompt_name = "v3_agent_analytical" if is_analytical else "v3_agent"

    system_content = get_system_prompt(version=prompt_name)
    history = state.context_messages or []
    agent_messages: list[ChatMessage] = [
        ChatMessage(role=Role.system, content=system_content),
        *[ChatMessage(role=Role(m.role.value), content=m.content or "") for m in history],
    ]

    # Step 1: inject exact entity names from the router so the agent uses correct strings
    # and knows which entities it must cover before calling report_findings.
    # Also inject metadata-backed years so the agent does not hallucinate fiscal year terms.

    # Build year list per entity from entity_manifest (real document metadata)
    _entity_years: dict[str, list[int]] = {}
    if state.scope_result and state.scope_result.entity_manifest:
        for item in state.scope_result.entity_manifest:
            years = sorted(
                {s["year"] for s in (item.doc_summaries or []) if s.get("year")},
                reverse=True,
            )
            if years:
                _entity_years[item.entity_name] = years

    _expected_entities: frozenset[str] = frozenset()
    if not is_analytical and state.scope_result and state.scope_result.per_entity_doc_ids:
        _expected_entities = frozenset(state.scope_result.per_entity_doc_ids.keys())
        lines: list[str] = []
        for name in sorted(_expected_entities):
            years = _entity_years.get(name)
            suffix = f" (available years: {', '.join(str(y) for y in years)})" if years else ""
            lines.append(f"- {name}{suffix}")

        agent_messages.append(
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
        # For analytical queries, inject available years so the agent grounds temporal references
        year_lines = [
            f"- {name}: {', '.join(str(y) for y in years)}"
            for name, years in sorted(_entity_years.items())
        ]
        agent_messages.append(
            ChatMessage(
                role=Role.user,
                content=(
                    "Available document years (use ONLY these in search queries — do not invent fiscal years):\n"
                    + "\n".join(year_lines)
                ),
            )
        )

    agent_messages.append(ChatMessage(role=Role.user, content=state.user_query_raw))

    chunk_registry: dict[UUID, RetrievedChunk] = {}
    # Globally unique S-labels across all searches in this request: tool results are
    # numbered S1..Sn continuously, and ref_registry maps each label back to its chunk
    # so labels the agent reports in findings can be resolved to UUIDs.
    ref_registry: dict[str, UUID] = {}
    next_ref = 1
    # Every chunk shown to the agent in tool results, regardless of registry admission —
    # used to force-admit cited chunks that the per-entity cap excluded.
    # Per-lookup grouping of the chunks each search first admitted to the registry,
    # used by the post-loop context-window cap (Step 10). Each search_documents call
    # gets a distinct lookup_id; capping is deferred to loop end because we cannot tell
    # whether more lookups will follow until the agent stops searching.
    lookup_chunks: dict[int, list[UUID]] = {}
    lookup_count = 0
    searched_entities: set[str] = set()
    token_spend = 0
    output_tokens_total = 0
    cost_usd_total = 0.0
    tool_calls_total = 0
    convergence_reason: Literal[
        "natural", "convergence", "iteration_cap", "budget_cap", "timeout"
    ] = "iteration_cap"
    agent_findings: AgentFindings | AnalyticalFindings | None = None

    lf = lf_client.get_client()
    iteration = 0

    for iteration in range(max_iterations):
        await add_event(redis_app, request_id, "agent_turn_started", {"iteration": iteration})
        logger.debug("agent_turn_started", extra={"request_id": request_id, "iteration": iteration})

        _turn_lf_stack = contextlib.ExitStack()
        if lf:
            _turn_lf_stack.enter_context(
                lf.start_as_current_observation(
                    as_type="span",
                    name=f"agent_turn_{iteration}",
                    input={
                        "iteration": iteration,
                        "messages": [
                            {"role": m.role.value, "content": (m.content or "")[:500]}
                            for m in agent_messages
                        ],
                    },
                )
            )
        new_chunks = 0
        turn: AssistantTurnResult | None = None

        try:
            turn = await asyncio.wait_for(
                llm.complete_with_tools(agent_messages, tools=tools),
                timeout=_AGENT_TURN_TIMEOUT,
            )
            if turn.stats:
                token_spend += turn.stats.input_tokens or 0
                output_tokens_total += turn.stats.output_tokens or 0
                cost_usd_total += turn.stats.cost_usd or 0.0
                if turn.stats.input_tokens:
                    LLM_TOKENS.labels("input", llm.model_id).inc(turn.stats.input_tokens)
                if turn.stats.output_tokens:
                    LLM_TOKENS.labels("output", llm.model_id).inc(turn.stats.output_tokens)
                if turn.stats.cached_input_tokens:
                    LLM_CACHE_HIT_TOKENS.labels(llm.model_id).inc(turn.stats.cached_input_tokens)
                if turn.stats.cost_usd:
                    LLM_COST.labels(llm.model_id).inc(turn.stats.cost_usd)

                # Step 23: log each tool-calling turn as a subrequest row
                if state.llm_request and state.llm_request.conversation_id is not None:
                    with contextlib.suppress(Exception):
                        await LLMRequestRepository(session).create_subrequest(
                            parent_request_id=state.llm_request.id,
                            conversation_id=state.llm_request.conversation_id,
                            user_id=state.llm_request.user_id,
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
                convergence_reason = "natural"
                break

            # Check for finalizer — terminates the loop
            finalizer_tc = next((tc for tc in turn.tool_calls if tc.name in _FINALIZER_NAMES), None)
            if finalizer_tc is not None:
                # Step 2: guard — reject report_findings if any expected entities were not searched
                if finalizer_tc.name == "report_findings" and _expected_entities:
                    missing = _expected_entities - searched_entities
                    if missing:
                        agent_messages.append(_assistant_msg_with_tool_calls([finalizer_tc]))
                        agent_messages.append(
                            ChatMessage(
                                role=Role.tool,
                                tool_call_id=finalizer_tc.id,
                                content=(
                                    f"report_findings rejected — missing searches for: "
                                    f"{', '.join(sorted(missing))}. "
                                    "Search each missing entity before calling report_findings."
                                ),
                            )
                        )
                        continue  # force another iteration

                _fin_lf_stack = contextlib.ExitStack()
                if lf:
                    try:
                        _fin_input = json.loads(finalizer_tc.arguments)
                    except Exception:
                        _fin_input = {"raw": finalizer_tc.arguments[:500]}
                    _fin_lf_stack.enter_context(
                        lf.start_as_current_observation(
                            as_type="span",
                            name=finalizer_tc.name,
                            input=_fin_input,
                        )
                    )
                try:
                    agent_findings = _resolve_finding_refs(
                        _parse_findings(finalizer_tc), ref_registry, request_id
                    )
                    AGENT_TOOL_CALLS.labels(finalizer_tc.name, "ok").inc()
                    # All chunks shown to the agent are admitted to the registry at search
                    # time; the post-loop per-lookup cap explicitly preserves cited chunks
                    # (see _finding_chunk_ids below), so no force-admit is needed here.
                    if lf:
                        if isinstance(agent_findings, AgentFindings):
                            _findings_summary = [
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
                                for f in agent_findings.findings
                            ]
                            lf.update_current_span(
                                output={
                                    "metric_requested": agent_findings.metric_requested,
                                    "comparison_op": agent_findings.comparison_op,
                                    "findings": _findings_summary,
                                },
                                metadata={
                                    "parse_ok": True,
                                    "findings_count": len(agent_findings.findings),
                                },
                            )
                        else:
                            lf.update_current_span(
                                output={
                                    "question": agent_findings.question,
                                    "conclusion": agent_findings.conclusion,
                                    "gaps": agent_findings.gaps,
                                    "observations": [
                                        {
                                            "claim": o.claim,
                                            "confidence": o.confidence,
                                            "evidence_chunks": o.evidence_chunks,
                                        }
                                        for o in agent_findings.observations
                                    ],
                                },
                                metadata={
                                    "parse_ok": True,
                                    "observations_count": len(agent_findings.observations),
                                },
                            )
                except Exception:
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
                        {
                            "entity": "__finalizer__",
                            "error": True,
                            "reason": "findings_parse_failed",
                        },
                    )
                finally:
                    _fin_lf_stack.close()
                convergence_reason = "natural"
                break

            search_tcs = turn.tool_calls

            tool_calls_total += len(turn.tool_calls)
            agent_messages.append(_assistant_msg_with_tool_calls(turn.tool_calls))

            async def _guarded_search(tc: ToolCallRef, _iter: int = iteration) -> _SearchResult:
                async with _search_sem:
                    return await _execute_search(
                        tc, state, session, reranker, redis_app, request_id, _iter
                    )

            results = await asyncio.gather(*[_guarded_search(tc) for tc in search_tcs])

            new_chunks = 0
            for tc, result in zip(search_tcs, results, strict=False):
                if result.rewrite_stats:
                    token_spend += result.rewrite_stats.input_tokens or 0
                    output_tokens_total += result.rewrite_stats.output_tokens or 0
                    cost_usd_total += result.rewrite_stats.cost_usd or 0.0
                entity_new = 0
                entity = json.loads(tc.arguments).get("entity", "")
                # Step 10: admit every chunk uncapped here; the per-lookup context-window
                # cap is applied once after the loop ends. Chunks are tracked by the lookup
                # that first admitted them so the cap can keep the top-N of each lookup
                # without discarding a single-lookup query's full set.
                lookup_id = lookup_count
                lookup_count += 1
                this_lookup: list[UUID] = []
                for chunk in result.chunks:
                    chunk_id = chunk.chunk_id
                    if chunk_id not in chunk_registry:
                        chunk_registry[chunk_id] = chunk
                        this_lookup.append(chunk_id)
                        new_chunks += 1
                        entity_new += 1
                lookup_chunks[lookup_id] = this_lookup
                if entity:
                    searched_entities.add(entity)
                logger.debug(
                    "tool_call_completed",
                    extra={
                        "request_id": request_id,
                        "iteration": iteration,
                        "entity": entity,
                        "chunks_returned": len(result.chunks),
                        "new_chunks_added": entity_new,
                    },
                )
                await add_event(
                    redis_app,
                    request_id,
                    "tool_call_completed",
                    {
                        "entity": entity,
                        "chunks_returned": len(result.chunks),
                        "new_chunks_added": entity_new,
                    },
                )
                # Assemble the tool-result context here (sequentially) so S-labels
                # continue across searches instead of restarting at S1 each time.
                if result.error_str is not None:
                    tool_content = result.error_str
                else:
                    ctx, _ = assemble_rag_context(
                        result.chunks,
                        result.payloads,
                        assume_unique=True,
                        ref_start=next_ref,
                    )
                    for item in ctx.items:
                        ref_registry[item.ref_id] = item.chunk_id
                    next_ref += len(ctx.items)
                    tool_content = ctx.formatted_context or "(no results)"
                agent_messages.append(
                    ChatMessage(
                        role=Role.tool,
                        tool_call_id=tc.id,
                        content=tool_content,
                    )
                )

            if new_chunks == 0:
                convergence_reason = "convergence"
                break

            if token_spend > token_budget:
                convergence_reason = "budget_cap"
                break

            agent_messages = _compress_history(agent_messages, keep_last_n_turns=2)

        except TimeoutError:
            logger.warning(
                "agent_turn_timeout",
                extra={
                    "request_id": request_id,
                    "iteration": iteration,
                    "timeout": _AGENT_TURN_TIMEOUT,
                },
            )
            convergence_reason = "timeout"
            break

        finally:
            if lf:
                lf.update_current_span(
                    output={
                        "tool_calls": [
                            {"name": tc.name, "arguments": tc.arguments}
                            for tc in (
                                turn.tool_calls if turn is not None and turn.tool_calls else []
                            )
                        ],
                        "convergence_reason": convergence_reason,
                    },
                    metadata={"token_spend_cumulative": token_spend, "new_chunks": new_chunks},
                )
            _turn_lf_stack.close()

    # Step 10: per-lookup context-window cap. A single lookup keeps its full result set
    # (≤ retrieval top_k, safe for the synthesis context). With more than one lookup the
    # combined volume can overflow the synthesis context, so each lookup is trimmed to its
    # top-N chunks (already in reranker order within lookup_chunks).
    if lookup_count > 1:
        keep_ids: set[UUID] = set()
        for cids in lookup_chunks.values():
            keep_ids.update(cids[:max_chunks_per_entity])
        # Never evict a chunk the agent cited in its findings — those must stay citable
        # in the synthesis context regardless of where they fell in the per-lookup ranking.
        if agent_findings is not None:
            for cid_str in _finding_chunk_ids(agent_findings):
                with contextlib.suppress(ValueError):
                    keep_ids.add(UUID(cid_str))
        if len(keep_ids) < len(chunk_registry):
            chunk_registry = {cid: chunk_registry[cid] for cid in keep_ids}

    await add_event(
        redis_app,
        request_id,
        "agent_synthesis_starting",
        {"total_chunks": len(chunk_registry), "iterations": iteration + 1},
    )

    return (
        chunk_registry,
        agent_findings,
        AgentLoopMeta(
            iterations=iteration + 1,
            tool_calls_total=tool_calls_total,
            convergence_reason=convergence_reason,
            input_tokens_total=token_spend,
            output_tokens_total=output_tokens_total,
            cost_usd_total=cost_usd_total,
        ),
    )
