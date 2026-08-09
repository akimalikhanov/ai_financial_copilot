"""The synthesis boundary: agent-loop output -> the context and findings synthesis reads.

Single home for the sequence `tasks.py` and `src.eval.pipeline_agent` previously each
re-implemented (and had drifted, see docs/stages/agentic_state_refactor_v2.md P0-5):
resolve findings -> inject stubs for entities the agent never searched -> normalize FX ->
select the evidence the findings actually cite -> assemble the RAG context -> render the
findings block -> concatenate. Both callers now collapse to a single `run_agent` call
(see `__init__.py`), which calls this after the loop.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from uuid import UUID

from src.observability.langfuse import get_client as lf_get_client
from src.schemas.agent_findings import AgentFindings, AnalyticalFindings, EntityFinding
from src.schemas.query_router import DocumentScopeResult
from src.schemas.retrieval import RAGContext
from src.services.chat.agent.evidence import EvidenceLedger
from src.services.chat.agent.processor import (
    ProcessedFindings,
    _render_findings_block,
    _render_observations_block,
    process_findings,
)
from src.services.chat.agent.state import AgentLoopMeta
from src.services.retrieval.context_assembler import assemble_rag_context


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
    """Defence-in-depth stub for every in-scope entity absent from the findings.

    Keyed on reported coverage, not ``searched_entities``: an entity that was searched but
    never landed a finding (P0-A — e.g. every citation failed ``resolve_refs``) would
    otherwise vanish from the findings block, the evidence selection and the answer with
    no trace, silently turning a 2-entity comparison into a 1-entity argmax.
    ``searched_entities`` still picks the reason, so a never-searched entity isn't
    mislabeled as searched-but-ungrounded (P1-0). Keying on reported coverage also means a
    model that self-reports "unavailable" without searching gets its own row and not a
    duplicate stub (P2-G).
    """
    if scope_result is None or not scope_result.per_entity_doc_ids:
        return findings
    reported = {f.entity for f in findings.findings}
    missing_stubs = tuple(
        EntityFinding(
            entity=name,
            available=False,
            reason=(
                "not searched by agent"
                if name not in agent_meta.searched_entities
                else "searched, but no value could be grounded in the retrieved excerpts"
            ),
        )
        for name in sorted(scope_result.per_entity_doc_ids.keys())
        if name not in reported
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
    evidence: EvidenceLedger,
    agent_findings: AgentFindings | AnalyticalFindings | None,
    agent_meta: AgentLoopMeta,
    scope_result: DocumentScopeResult | None,
    requested_currency: str | None,
    max_chunks_per_entity: int,
) -> AgentRunResult:
    ordered = evidence.ordered_chunks()
    # The fallback pool is what the model could actually read: falling back to excerpts
    # it never saw would let synthesis cite text no reasoning was ever grounded in.
    fallback = evidence.rendered_chunks()[:max_chunks_per_entity]

    findings = agent_findings
    processed: ProcessedFindings | None = None

    if findings is not None:
        if isinstance(findings, AgentFindings):
            findings = _inject_unsearched_stubs(findings, agent_meta, scope_result)

            # No dedicated span: this wraps a single `process_findings` call with no
            # sub-structure of its own (the interesting nested work is `fx_conversion`,
            # inside `process_findings`), so its input/output land on the enclosing
            # `agent_loop` span instead of paying for another hop with no new information.
            cited_chunk_ids = {c for f in findings.findings for c in (f.source_chunks or [])}
            processed = await process_findings(
                findings,
                requested_currency=requested_currency,
                chunk_texts=evidence.texts_for(cited_chunk_ids),
            )
            lf = lf_get_client()
            if lf:
                with contextlib.suppress(Exception):
                    lf.update_current_span(
                        metadata={
                            "findings_processor": {
                                "input": {
                                    "metric_requested": findings.metric_requested,
                                    "comparison_op": findings.comparison_op,
                                    "findings": [f.model_dump() for f in findings.findings],
                                },
                                "output": {
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
                                            "number_grounding": nf.number_grounding.value,
                                        }
                                        for nf in processed.findings
                                    ],
                                },
                            }
                        }
                    )

        # Narrow the synthesis context to the chunks the agent actually cited in its
        # findings — those are the evidence it reasoned over. When findings cite nothing
        # (e.g. a weak tool model that omits source_chunks), fall back to the chunks the
        # model was actually shown rather than starving synthesis.
        cited_ids = _cited_chunk_ids(findings)
        synthesis_chunks = (
            [c for c in ordered if c.chunk_id in cited_ids] if cited_ids else fallback
        )
    else:
        synthesis_chunks = fallback

    # Payloads were cached (already sanitized) when the loop rendered these chunks — no
    # second hydration, and the re-scan below is a no-op over sanitized text (D2).
    payloads = evidence.payloads_for(c.chunk_id for c in synthesis_chunks)
    synthesis_chunks = [c for c in synthesis_chunks if c.chunk_id in payloads]
    rag_context, _ = assemble_rag_context(synthesis_chunks, payloads, assume_unique=True)

    # Pick the renderer by findings type: analytical runs have no values to FX-normalize,
    # so they skip `process_findings` entirely and render their observations directly.
    if isinstance(findings, AnalyticalFindings):
        findings_block = _render_observations_block(findings, rag_context)
    elif processed is not None:
        findings_block = _render_findings_block(processed, rag_context)
    else:
        findings_block = None

    synthesis_context = (
        findings_block + "\n\n" + (rag_context.formatted_context or "")
        if findings_block is not None
        else rag_context.formatted_context or "(No document context.)"
    )

    return AgentRunResult(
        rag_context=rag_context,
        synthesis_context=synthesis_context,
        findings=findings,
        processed=processed,
        meta=agent_meta,
    )
