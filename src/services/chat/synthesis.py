"""The synthesis boundary: agent-loop output -> the context and findings synthesis reads.

Single home for the sequence `tasks.py` and `src.eval.pipeline_agent` previously each
re-implemented (and had drifted, see docs/stages/agentic_state_refactor_v2.md P0-5):
resolve findings -> inject stubs for entities the agent never searched -> normalize FX ->
select the evidence the findings actually cite -> assemble the RAG context -> render the
findings block -> concatenate. Both callers now collapse to a single `run_synthesis` call.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from src.observability import langfuse as lf_client
from src.schemas.agent_findings import AgentFindings, AnalyticalFindings, EntityFinding
from src.schemas.query_router import DocumentScopeResult
from src.schemas.retrieval import RAGContext, RetrievedChunk
from src.services.chat.agent_loop import AgentLoopMeta, _order_chunks
from src.services.chat.findings_processor import (
    ProcessedFindings,
    _render_findings_block,
    _render_observations_block,
    process_findings,
)
from src.services.retrieval.context_assembler import assemble_rag_context
from src.services.retrieval.payload_hydrator import get_chunk_prompt_payloads


@dataclass(frozen=True)
class AgentRunResult:
    rag_context: RAGContext
    synthesis_context: str
    findings: AgentFindings | AnalyticalFindings | None
    processed: ProcessedFindings | None
    meta: AgentLoopMeta


def _inject_unsearched_stubs(
    findings: AgentFindings,
    agent_meta: AgentLoopMeta,
    scope_result: DocumentScopeResult | None,
) -> AgentFindings:
    """Defence-in-depth stub for entities the agent never searched.

    Uses ``searched_entities`` (what the loop actually did), not reported coverage, so an
    entity that was searched but simply not reported in the finalizer isn't mislabeled
    "not searched by agent" (P1-0).
    """
    if scope_result is None or not scope_result.per_entity_doc_ids:
        return findings
    missing_stubs = tuple(
        EntityFinding(entity=name, available=False, reason="not searched by agent")
        for name in sorted(scope_result.per_entity_doc_ids.keys())
        if name not in agent_meta.searched_entities
    )
    if not missing_stubs:
        return findings
    return findings.model_copy(update={"findings": findings.findings + missing_stubs})


def _cited_chunk_ids(findings: AgentFindings | AnalyticalFindings) -> set[UUID]:
    cited_ids: set[UUID] = set()
    if isinstance(findings, AgentFindings):
        for f in findings.findings:
            for chunk_id in f.source_chunks or []:
                with contextlib.suppress(ValueError):
                    cited_ids.add(UUID(chunk_id))
    else:
        for obs in findings.observations:
            for chunk_id in (*obs.evidence_chunks, *(obs.refuted_by or [])):
                with contextlib.suppress(ValueError):
                    cited_ids.add(UUID(chunk_id))
    return cited_ids


async def run_synthesis(
    chunk_registry: dict[UUID, RetrievedChunk],
    agent_findings: AgentFindings | AnalyticalFindings | None,
    agent_meta: AgentLoopMeta,
    scope_result: DocumentScopeResult | None,
    requested_currency: str | None,
    session: AsyncSession,
) -> AgentRunResult:
    ordered = _order_chunks(chunk_registry)
    lf = lf_client.get_client()

    findings = agent_findings
    processed: ProcessedFindings | None = None

    if findings is not None:
        if isinstance(findings, AgentFindings):
            findings = _inject_unsearched_stubs(findings, agent_meta, scope_result)

        _lf_stack = contextlib.ExitStack()
        if lf:
            _lf_stack.enter_context(
                lf.start_as_current_observation(
                    as_type="span",
                    name="findings_processor",
                    input={
                        "type": type(findings).__name__,
                        "metric_requested": getattr(findings, "metric_requested", None),
                        "comparison_op": getattr(findings, "comparison_op", None),
                        "findings": [f.model_dump() for f in findings.findings]
                        if isinstance(findings, AgentFindings)
                        else None,
                    },
                )
            )
        try:
            processed = await process_findings(findings, requested_currency=requested_currency)
            if lf:
                lf.update_current_span(
                    output={
                        "currency_converted": processed.currency_converted,
                        "answer_entity": processed.answer_entity,
                        "fx_rates_used": processed.fx_rates_used,
                        "answer_note": processed.answer_note,
                        "comparison_op": processed.comparison_op,
                        "findings": [
                            {
                                "entity": nf.finding.entity,
                                "normalized_value": nf.normalized_value,
                                "fx_rate": nf.fx_rate,
                                "native_value": nf.finding.value,
                                "currency": nf.finding.currency,
                                "unit": nf.finding.unit,
                                "period_end": nf.finding.period_end,
                                "available": nf.finding.available,
                            }
                            for nf in processed.findings
                        ],
                    },
                )
        finally:
            _lf_stack.close()

        # Narrow the synthesis context to the chunks the agent actually cited in its
        # findings — those are the evidence it reasoned over, and the registry is
        # already volume-capped per lookup. When findings cite nothing (e.g. a weak
        # tool model that omits source_chunks), fall back to the full capped registry
        # rather than starving synthesis.
        cited_ids = _cited_chunk_ids(findings)
        synthesis_chunks = [c for c in ordered if c.chunk_id in cited_ids] if cited_ids else ordered
    else:
        synthesis_chunks = ordered

    chunk_ids = [c.chunk_id for c in synthesis_chunks]
    payloads = await get_chunk_prompt_payloads(session, chunk_ids)
    rag_context, _ = assemble_rag_context(synthesis_chunks, payloads, assume_unique=True)

    if processed is not None:
        if processed.analytical_findings is not None:
            findings_block = _render_observations_block(
                processed.analytical_findings, rag_context=rag_context
            )
        else:
            findings_block = _render_findings_block(processed, rag_context=rag_context)
        synthesis_context = findings_block + "\n\n" + (rag_context.formatted_context or "")
    else:
        synthesis_context = rag_context.formatted_context or "(No document context.)"

    return AgentRunResult(
        rag_context=rag_context,
        synthesis_context=synthesis_context,
        findings=findings,
        processed=processed,
        meta=agent_meta,
    )
