"""EvidenceLedger — the agent's durable, chunk-level record for one run.

Holds the run's chunk/ref bookkeeping with a single invariant: every S-label the transcript
has shown resolves through `resolve_refs`. The transcript is append-only, so every label
also stays readable on screen; each chunk is rendered once per run.

Single writer: `admit`/`assign_labels` are called only on the loop task, only in the
post-`gather` reduce. Search handlers are pure and never receive a ledger reference, so
`next_ref` label allocation stays deterministic under concurrent search.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from dataclasses import replace as dc_replace
from uuid import UUID

from src.schemas.retrieval import (
    REF_PLACEHOLDER,
    ChunkPromptPayload,
    ContextItem,
    RAGContext,
    RetrievedChunk,
)
from src.services.retrieval.context_assembler import assemble_rag_context
from src.utils.config import get_agent_shown_heading_chars


@dataclass
class EvidenceRecord:
    chunk: RetrievedChunk
    best_score: float = 0.0
    ref_id: str | None = None


class EvidenceLedger:
    def __init__(self) -> None:
        self._records: dict[UUID, EvidenceRecord] = {}
        self._ref_registry: dict[str, UUID] = {}
        # Every chunk ever labelled, i.e. rendered into the transcript.
        self._items: dict[UUID, ContextItem] = {}
        # Sanitized payloads as rendered, so synthesis re-assembles without a second
        # DB hydration or a second injection scan over the same text.
        self._payloads: dict[UUID, ChunkPromptPayload] = {}
        self._next_ref = 1

    def admit(self, chunks: Sequence[RetrievedChunk]) -> int:
        """Register the chunks one search returned. Returns the count newly admitted."""
        new_count = 0
        for chunk in chunks:
            record = self._records.get(chunk.chunk_id)
            if record is None:
                self._records[chunk.chunk_id] = EvidenceRecord(
                    chunk=chunk, best_score=chunk.score or 0.0
                )
                new_count += 1
            else:
                record.best_score = max(record.best_score, chunk.score or 0.0)
        return new_count

    def shown_before(self, chunks: Sequence[RetrievedChunk]) -> list[str]:
        """`Sn heading` for each of `chunks` an earlier search already rendered, in the
        given order. Call before `assign_labels`, which labels the rest."""
        max_chars = get_agent_shown_heading_chars()
        out: list[str] = []
        for chunk in chunks:
            item = self._items.get(chunk.chunk_id)
            if item is None:
                continue
            heading = chunk.heading_trail[-1] if chunk.heading_trail else ""
            if len(heading) > max_chars:
                heading = heading[:max_chars].rstrip() + "…"
            out.append(f"{item.ref_id} {heading}".rstrip())
        return out

    def assign_labels(
        self,
        chunks: Sequence[RetrievedChunk],
        payloads: dict[UUID, ChunkPromptPayload],
    ) -> RAGContext:
        """Assemble one tool result's RAGContext from the chunks never labelled before,
        numbering S-labels globally across the request so labels never restart at S1
        between searches.

        A chunk an earlier search labelled keeps its one label and is not rendered again:
        the transcript is append-only, so it is still on screen. Re-surfacing updates
        provenance in `admit`, never mints a second label.
        """
        fresh = [c for c in chunks if c.chunk_id not in self._items]
        ctx, _ = assemble_rag_context(fresh, payloads, assume_unique=True, ref_start=self._next_ref)
        for item in ctx.items:
            self._ref_registry[item.ref_id] = item.chunk_id
            self._items[item.chunk_id] = item
            payload = payloads.get(item.chunk_id)
            if payload is not None:
                # Store the post-scan text: synthesis re-assembly then re-runs the
                # injection scan over already-sanitized text, a provable no-op.
                self._payloads[item.chunk_id] = dc_replace(
                    payload, prompt_text=item.prompt_text.replace(item.ref_id, REF_PLACEHOLDER, 1)
                )
            record = self._records.get(item.chunk_id)
            if record is not None:
                record.ref_id = item.ref_id
        self._next_ref += len(ctx.items)
        return ctx

    def resolve_refs(self, refs: list[str] | None) -> tuple[list[str], list[str]]:
        """Map agent-reported S-labels (or already-UUID refs) to chunk-UUID strings.

        Returns (resolved, unresolved).
        """
        resolved: list[str] = []
        unresolved: list[str] = []
        for ref in refs or []:
            candidate = ref.strip()
            try:
                UUID(candidate)
            except ValueError:
                chunk_id = self._ref_registry.get(candidate.upper())
                if chunk_id is not None:
                    resolved.append(str(chunk_id))
                else:
                    unresolved.append(candidate)
            else:
                resolved.append(candidate)
        return resolved, unresolved

    def payloads_for(self, chunk_ids: Iterable[UUID]) -> dict[UUID, ChunkPromptPayload]:
        """Sanitized payloads cached at render time — no second hydration. Every
        chunk synthesis can select was rendered, so a cached payload always exists."""
        return {cid: self._payloads[cid] for cid in chunk_ids if cid in self._payloads}

    def texts_for(self, chunk_ids: Iterable[str]) -> dict[str, str]:
        """Sanitized rendered text keyed by chunk-UUID *string* — the form findings carry
        after refs are resolved to UUIDs. Number grounding reads this; the payloads are
        already cached, so this is zero-I/O like `payloads_for`."""
        out: dict[str, str] = {}
        for raw in chunk_ids:
            with contextlib.suppress(ValueError):
                payload = self._payloads.get(UUID(raw))
                if payload is not None:
                    out[raw] = payload.prompt_text
        return out

    def ordered_chunks(self) -> list[RetrievedChunk]:
        """Every admitted chunk, best-scoring first, globally. `assemble_rag_context`
        assigns S-labels in input order, so this is what makes S1 the best chunk of the
        whole run."""
        return sorted((r.chunk for r in self._records.values()), key=lambda c: -(c.score or 0))

    def labelled_chunks(self) -> list[RetrievedChunk]:
        """Every chunk the model was shown, best-scoring first — the only defensible
        candidates for a synthesis fallback (falling back to excerpts the model never saw
        is not)."""
        return sorted(
            (self._records[cid].chunk for cid in self._items if cid in self._records),
            key=lambda c: -(c.score or 0),
        )

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, chunk_id: UUID) -> bool:
        return chunk_id in self._records
