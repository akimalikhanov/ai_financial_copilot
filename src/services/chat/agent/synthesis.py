"""The synthesis boundary: agent-loop output -> the context and findings synthesis reads.

Single home for the sequence both `tasks.py` and `src.eval.pipeline_agent` need: normalize
FX -> select the evidence the findings actually cite -> assemble the RAG context -> render
the findings block -> concatenate. Both callers reach it through one `run_agent` call (see
`__init__.py`), which calls this after the loop.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from uuid import UUID

from src.observability.langfuse import get_client as lf_get_client
from src.schemas.agent_findings import AgentFindings
from src.schemas.retrieval import RAGContext
from src.services.chat.agent.evidence import EvidenceLedger
from src.services.chat.agent.processor import (
    ProcessedFindings,
    _render_findings_block,
    process_findings,
)
from src.services.chat.agent.state import AgentLoopMeta
from src.services.retrieval.context_assembler import assemble_rag_context


@dataclass(frozen=True)
class AgentRunResult:
    rag_context: RAGContext
    synthesis_context: str
    findings: AgentFindings | None
    processed: ProcessedFindings | None
    meta: AgentLoopMeta
    # The rendered block alone, without the excerpts — carried to a follow-up turn so it
    # can be answered without re-retrieving (see followup_direct_answer_plan.md).
    findings_block: str | None = None


# Prepended to the synthesis context when every search errored. Without it a dead
# retrieval backend is indistinguishable from an empty corpus: the context collapses to
# "(No document context.)" and synthesis reports "not found in the uploaded documents",
# telling the user their documents lack a fact they may well contain.
RETRIEVAL_UNAVAILABLE_BANNER = (
    "[RETRIEVAL UNAVAILABLE]\n"
    "Document search failed for this request — every search returned a backend error, so "
    "NO excerpts could be retrieved. This is a temporary system failure, NOT evidence that "
    "the documents lack the requested information.\n"
    "Tell the user document search is temporarily unavailable and ask them to retry. Do NOT "
    "state or imply the information was not found in the documents, and do NOT output the "
    "`N/A` not-found marker."
)


def _cited_chunk_ids(findings: AgentFindings) -> set[UUID]:
    cited_ids: set[UUID] = set()
    for f in findings.findings:
        for chunk_id in f.evidence:
            with contextlib.suppress(ValueError):
                cited_ids.add(UUID(chunk_id))
    return cited_ids


async def run_synthesis(
    evidence: EvidenceLedger,
    findings: AgentFindings | None,
    agent_meta: AgentLoopMeta,
    requested_currency: str | None,
    max_chunks_per_entity: int,
) -> AgentRunResult:
    ordered = evidence.ordered_chunks()
    # The fallback pool is what the model could actually read: falling back to excerpts
    # it never saw would let synthesis cite text no reasoning was ever grounded in.
    fallback = evidence.labelled_chunks()[:max_chunks_per_entity]

    processed: ProcessedFindings | None = None

    if findings is not None:
        # No dedicated span: this wraps a single `process_findings` call with no
        # sub-structure of its own (the interesting nested work is `fx_conversion`, inside
        # `process_findings`), so its output lands on the enclosing `agent_loop` span
        # instead of paying for another hop with no new information.
        processed = await process_findings(
            findings,
            requested_currency=requested_currency,
            chunk_texts=evidence.texts_for({c for f in findings.findings for c in f.evidence}),
        )
        lf = lf_get_client()
        if lf:
            with contextlib.suppress(Exception):
                lf.update_current_span(
                    metadata={
                        "findings_processor": {
                            "currency_converted": processed.currency_converted,
                            "answer_entity": processed.answer_entity,
                            "fx_rates_used": processed.fx_rates_used,
                            "answer_note": processed.answer_note,
                            "figures": [
                                {
                                    "key": n.key,
                                    **n.figure.model_dump(),
                                    "normalized_amount": n.normalized_amount,
                                    "fx_rate": n.fx_rate,
                                    "number_grounding": n.number_grounding.value,
                                }
                                for n in processed.figures
                            ],
                        }
                    }
                )

        # Narrow the synthesis context to the chunks the agent actually cited in its
        # findings — those are the evidence it reasoned over. When findings cite nothing
        # (e.g. a weak tool model that omits evidence), fall back to the chunks the model
        # was actually shown rather than starving synthesis.
        cited_ids = _cited_chunk_ids(findings)
        synthesis_chunks = (
            [c for c in ordered if c.chunk_id in cited_ids] if cited_ids else fallback
        )
    else:
        synthesis_chunks = fallback

    # Payloads were cached (already sanitized) when the loop rendered these chunks — no
    # second hydration, and the re-scan below is a no-op over sanitized text.
    payloads = evidence.payloads_for(c.chunk_id for c in synthesis_chunks)
    synthesis_chunks = [c for c in synthesis_chunks if c.chunk_id in payloads]
    rag_context, _ = assemble_rag_context(synthesis_chunks, payloads, assume_unique=True)

    findings_block = (
        _render_findings_block(processed, rag_context) if processed is not None else None
    )

    synthesis_context = (
        findings_block + "\n\n" + (rag_context.formatted_context or "")
        if findings_block is not None
        else rag_context.formatted_context or "(No document context.)"
    )

    # Leads the context so the instruction is read before any (possibly empty) evidence.
    # Whatever partial findings a pre-failure turn landed still follow it — the banner
    # reframes an absence of excerpts, it does not discard evidence already in hand.
    if agent_meta.convergence_reason == "search_unavailable":
        synthesis_context = RETRIEVAL_UNAVAILABLE_BANNER + "\n\n" + synthesis_context

    return AgentRunResult(
        rag_context=rag_context,
        synthesis_context=synthesis_context,
        findings=findings,
        processed=processed,
        meta=agent_meta,
        findings_block=findings_block,
    )
