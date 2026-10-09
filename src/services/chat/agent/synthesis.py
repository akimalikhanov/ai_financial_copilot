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
from src.observability.langfuse import mark_current as lf_mark_current
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

# Placed right before the excerpts when no finding cites any, so the fallback excerpts are
# not mistaken for the vetted evidence a finding would have pointed to.
UNCITED_EXCERPTS_NOTICE = (
    "[EXCERPTS NOT BACKED BY FINDINGS]\n"
    "No finding cites the excerpts below. They are what the search agent read, not "
    "conclusions it reached. Use only what an excerpt states explicitly, cite it, and do not "
    "present it as a confirmed finding. A finding marked [not disclosed] still stands."
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
    fallback_max_chunks: int,
    mentions: dict[str, str] | None = None,
) -> AgentRunResult:
    # Both pools are what the model could actually read: citing or falling back to excerpts
    # it never saw would let synthesis cite text no reasoning was ever grounded in.
    labelled = evidence.labelled_chunks()
    # Selected round-robin across searches, then score-ordered like the cited pool, so S1
    # is still the strongest excerpt (the confidence badge reads items[0].score).
    fallback = sorted(evidence.fallback_chunks(fallback_max_chunks), key=lambda c: -(c.score or 0))

    processed: ProcessedFindings | None = None
    cited_ids: set[UUID] = set()

    if findings is not None:
        cited_ids = _cited_chunk_ids(findings)
        # No dedicated span: this wraps a single `process_findings` call with no
        # sub-structure of its own (the interesting nested work is `fx_conversion`, inside
        # `process_findings`), so its output lands on the enclosing `agent_loop` span
        # instead of paying for another hop with no new information.
        processed = await process_findings(
            findings,
            requested_currency=requested_currency,
            # Keyed by `str(UUID)`, the form `resolve_refs` gives finding evidence.
            chunk_texts={
                str(cid): p.prompt_text for cid, p in evidence.payloads_for(cited_ids).items()
            },
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

    # Narrow the synthesis context to the chunks the agent actually cited in its findings —
    # those are the evidence it reasoned over. When findings cite nothing (no report, only
    # stated negatives, or only unresolved lines), fall back to the chunks the model was
    # actually shown rather than starving synthesis.
    synthesis_chunks = [c for c in labelled if c.chunk_id in cited_ids] if cited_ids else fallback

    # Payloads were cached (already sanitized) when the loop rendered these chunks — no
    # second hydration, and the re-scan below is a no-op over sanitized text.
    payloads = evidence.payloads_for(c.chunk_id for c in synthesis_chunks)
    rag_context, _ = assemble_rag_context(synthesis_chunks, payloads, assume_unique=True)

    findings_block = (
        _render_findings_block(processed, rag_context, mentions) if processed is not None else None
    )

    excerpts = rag_context.formatted_context or ""
    if not cited_ids and rag_context.items:
        excerpts = UNCITED_EXCERPTS_NOTICE + "\n\n" + excerpts
        reason = "no_findings" if findings is None else "no_citations"
        lf_mark_current("WARNING", f"synthesis fell back to uncited excerpts ({reason})")
        lf = lf_get_client()
        if lf:
            with contextlib.suppress(Exception):
                lf.update_current_span(
                    metadata={
                        "synthesis_fallback": {
                            "reason": reason,
                            "excerpts": len(rag_context.items),
                            "shown": len(labelled),
                        }
                    }
                )

    synthesis_context = (
        findings_block + "\n\n" + excerpts
        if findings_block is not None
        else excerpts or "(No document context.)"
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
