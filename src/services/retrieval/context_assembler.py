"""Pure context assembly: dedup, budget, format chunks with citations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from uuid import UUID

from src.schemas.retrieval import (
    REF_PLACEHOLDER,
    ChunkPromptPayload,
    Citation,
    ContextItem,
    DroppedChunk,
    FlaggedChunk,
    RAGContext,
    RetrievedChunk,
)
from src.services.security.injection_detector import InjectionSignal, scan_retrieved_chunk
from src.utils.config import get_injection_scan_chunks_enabled


def _dedup_chunks(chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
    seen: set[UUID] = set()
    out: list[RetrievedChunk] = []
    for c in chunks:
        if c.chunk_id not in seen:
            seen.add(c.chunk_id)
            out.append(c)
    return out


def _wrap_excerpt(ref_id: str, source_doc: str, flagged: bool, text: str) -> str:
    flagged_attr = "true" if flagged else "false"
    return (
        f'<retrieved_excerpt id="{ref_id}" source_doc="{source_doc}" flagged="{flagged_attr}">\n'
        f"{text}\n"
        f"</retrieved_excerpt>"
    )


@dataclass
class AssemblyGuardrails:
    """Guardrail side-effects produced during context assembly."""

    dropped: list[DroppedChunk] = field(default_factory=list)
    flagged: list[FlaggedChunk] = field(default_factory=list)


def assemble_rag_context(
    chunks: Sequence[RetrievedChunk],
    payloads: Mapping[UUID, ChunkPromptPayload],
    *,
    assume_unique: bool = True,
) -> tuple[RAGContext, AssemblyGuardrails]:
    """Build RAGContext from reranked chunks and hydrated payloads.

    Returns (RAGContext, AssemblyGuardrails). Blocked chunks are dropped;
    flagged chunks are included but marked with flagged="true" in the XML wrapper.
    """
    if not assume_unique:
        selected = _dedup_chunks(chunks)
    else:
        selected = list(chunks)

    guardrails = AssemblyGuardrails()
    items: list[ContextItem] = []
    blocks: list[str] = []
    ref_counter = 1

    for chunk in selected:
        payload = payloads.get(chunk.chunk_id)
        if payload is None:
            raise ValueError(f"Missing ChunkPromptPayload for chunk_id={chunk.chunk_id}")

        # Scan the body only — prompt_text includes a header line "[Sn | doc | p.X]"
        # that adds benign tokens and dilutes instructional_density scores.
        # Skipped when INJECTION_SCAN_CHUNKS=false; chunk is treated as clean.
        if get_injection_scan_chunks_enabled():
            body = (
                payload.prompt_text.split("\n", 1)[1]
                if "\n" in payload.prompt_text
                else payload.prompt_text
            )
            signal = scan_retrieved_chunk(body)
        else:
            signal = InjectionSignal(score=0, severity="clean", sanitized_text=payload.prompt_text)

        if signal.severity == "block":
            guardrails.dropped.append(
                DroppedChunk(
                    chunk_id=str(chunk.chunk_id),
                    matched_rules=signal.matched_rules,
                    score=signal.score,
                )
            )
            continue

        flagged = signal.severity == "flag"
        if flagged:
            guardrails.flagged.append(
                FlaggedChunk(
                    chunk_id=str(chunk.chunk_id),
                    matched_rules=signal.matched_rules,
                    score=signal.score,
                )
            )

        ref_id = f"S{ref_counter}"
        ref_counter += 1
        prompt_text = signal.sanitized_text.replace(REF_PLACEHOLDER, ref_id)

        citation = Citation(
            ref_id=ref_id,
            ref_index=ref_counter - 1,
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            document_name=payload.document_name,
            filename=None,
            page_numbers=payload.page_numbers,
            heading_path=payload.heading_trail,
            snippet=payload.snippet,
        )

        items.append(
            ContextItem(
                ref_id=ref_id,
                chunk_id=chunk.chunk_id,
                score=chunk.score,
                prompt_text=prompt_text,
                citation=citation,
                provenance=payload.provenance,
            )
        )
        blocks.append(_wrap_excerpt(ref_id, payload.document_name, flagged, prompt_text))

    return (
        RAGContext(
            formatted_context="\n\n".join(blocks),
            items=tuple(items),
            chunk_count=len(items),
        ),
        guardrails,
    )
