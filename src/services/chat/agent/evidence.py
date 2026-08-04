"""EvidenceLedger — the agent's durable, chunk-level record for one run.

Absorbs the old `run_agent_loop`'s chunk/ref bookkeeping (chunk_registry, ref_registry,
next_ref) into one object with a single invariant: for every S-label the transcript has
ever shown, `resolve_refs` can still resolve it, regardless of what the transcript later
drops (Contract C1).

Contract C2 — single writer: `admit`/`assign_labels` are called only on the loop task,
only in the post-`gather` reduce. Search handlers are pure and never receive a ledger
reference, so `next_ref` label allocation stays deterministic under concurrent search.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from dataclasses import replace as dc_replace
from uuid import UUID

from src.observability.metrics import REVIVED_CHUNKS_PER_TURN
from src.schemas.retrieval import (
    REF_PLACEHOLDER,
    ChunkPromptPayload,
    ContextItem,
    RAGContext,
    RetrievedChunk,
)
from src.services.retrieval.context_assembler import assemble_rag_context, wrap_excerpt

# Per-turn cap on revivals, so a search re-returning a large evicted set cannot
# reinflate the transcript compaction just shrank.
_MAX_REVIVALS_PER_TURN = 3


def _wrap_excerpt_for(item: ContextItem, payloads: dict[UUID, ChunkPromptPayload]) -> str:
    """The excerpt exactly as rendered into the transcript, so a revived block carries its
    `id="Sn"` tag — `_LABEL_RE` must be able to re-evict it on a later compaction."""
    payload = payloads.get(item.chunk_id)
    doc_name = payload.document_name if payload else ""
    return wrap_excerpt(item.ref_id, doc_name, False, item.prompt_text)


@dataclass
class EvidenceRecord:
    chunk: RetrievedChunk
    best_score: float = 0.0
    ref_id: str | None = None


class EvidenceLedger:
    def __init__(self) -> None:
        self._records: dict[UUID, EvidenceRecord] = {}
        self._ref_registry: dict[str, UUID] = {}
        # Every chunk ever labelled, with the wrapped excerpt exactly as it was rendered.
        self._items: dict[UUID, ContextItem] = {}
        self._excerpts: dict[UUID, str] = {}
        # Sanitized payloads as rendered, so synthesis re-assembles without a second
        # DB hydration or a second injection scan over the same text (D2).
        self._payloads: dict[UUID, ChunkPromptPayload] = {}
        # Subset of _items currently visible in the model's transcript. Compaction
        # removes from here (via mark_evicted); a re-returned chunk is re-rendered.
        self._rendered: set[UUID] = set()
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

    def assign_labels(
        self,
        chunks: Sequence[RetrievedChunk],
        payloads: dict[UUID, ChunkPromptPayload],
        max_revivals: int = _MAX_REVIVALS_PER_TURN,
    ) -> RAGContext:
        """Assemble one tool result's RAGContext, numbering S-labels globally across the
        request so labels never restart at S1 between searches.

        A chunk already labelled — by an earlier search this run, or seeded from a prior
        turn via `from_carryover` — keeps its one stable label and is not re-rendered:
        re-surfacing updates provenance in `admit`, never mints a second S-label. This is
        the dedup step 11's carried-evidence seeding depends on.

        A chunk that *was* rendered but has since been evicted by compaction, and is now
        re-returned by a later search, is **revived**: re-emitted verbatim under its
        original label, minting no new ref. Without this it is readable in neither the
        transcript nor a fresh tool result, while `resolve_refs` still resolves it — so a
        claim could be grounded on text the model never actually read.
        """
        fresh = [c for c in chunks if c.chunk_id not in self._items]
        revived = [
            c for c in chunks if c.chunk_id in self._items and c.chunk_id not in self._rendered
        ]
        # Highest-scoring first, so a capped turn revives the most relevant evidence.
        revived.sort(key=lambda c: -(c.score or 0.0))
        revived = revived[:max_revivals]

        ctx, _ = assemble_rag_context(fresh, payloads, assume_unique=True, ref_start=self._next_ref)
        for item in ctx.items:
            self._ref_registry[item.ref_id] = item.chunk_id
            self._items[item.chunk_id] = item
            self._excerpts[item.chunk_id] = _wrap_excerpt_for(item, payloads)
            self._rendered.add(item.chunk_id)
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

        if not revived:
            return ctx

        REVIVED_CHUNKS_PER_TURN.observe(len(revived))
        for chunk in revived:
            self._rendered.add(chunk.chunk_id)
        revived_blocks = [self._excerpts[c.chunk_id] for c in revived]
        revived_items = tuple(self._items[c.chunk_id] for c in revived)
        blocks = (
            [*revived_blocks, ctx.formatted_context] if ctx.formatted_context else revived_blocks
        )
        return RAGContext(
            formatted_context="\n\n".join(b for b in blocks if b),
            items=(*revived_items, *ctx.items),
            chunk_count=len(revived_items) + ctx.chunk_count,
        )

    def mark_evicted(self, ref_ids: Iterable[str]) -> None:
        """Compaction dropped these labels from the model's view. Their chunks stay in
        `_items` (still resolvable, Contract C1) but leave `_rendered`, so a later search
        that re-returns one revives it rather than silently assuming it is still readable.
        """
        for ref_id in ref_ids:
            chunk_id = self._ref_registry.get(ref_id.strip().upper())
            if chunk_id is not None:
                self._rendered.discard(chunk_id)

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
        """Sanitized payloads cached at render time — no second hydration (D2). Every
        chunk synthesis can select was rendered, so a cached payload always exists."""
        return {cid: self._payloads[cid] for cid in chunk_ids if cid in self._payloads}

    def ordered_chunks(self) -> list[RetrievedChunk]:
        """Every admitted chunk, best-scoring first, globally. `assemble_rag_context`
        assigns S-labels in input order, so this is what makes S1 the best chunk of the
        whole run."""
        return sorted((r.chunk for r in self._records.values()), key=lambda c: -(c.score or 0))

    def rendered_chunks(self) -> list[RetrievedChunk]:
        """Chunks currently readable in the model's view — the only defensible candidates
        for a synthesis fallback (falling back to excerpts the model never saw is not)."""
        return sorted(
            (self._records[cid].chunk for cid in self._rendered if cid in self._records),
            key=lambda c: -(c.score or 0),
        )

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, chunk_id: UUID) -> bool:
        return chunk_id in self._records
