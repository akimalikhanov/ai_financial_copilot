"""EvidenceLedger — the agent's durable, chunk-level record for one run.

Absorbs six current locals from the old `run_agent_loop` (chunk_registry, ref_registry,
next_ref, lookup_chunks, lookup_count, ever_cited_chunk_ids) into one object with a
single invariant: for every S-label the transcript has ever shown, `resolve_refs` can
still resolve it, regardless of what the transcript later drops (Contract C1).

Contract C2 — single writer: `admit`/`assign_labels` are called only on the loop task,
only in the post-`gather` reduce. Search handlers are pure and never receive a ledger
reference, so `next_ref` label allocation stays deterministic under concurrent search.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from uuid import UUID

from src.schemas.retrieval import ChunkPromptPayload, RAGContext, RetrievedChunk
from src.services.retrieval.context_assembler import assemble_rag_context


@dataclass
class EvidenceRecord:
    chunk: RetrievedChunk
    first_seen_iteration: int
    seen_in_lookups: set[int] = field(default_factory=set)
    best_score: float = 0.0
    ref_id: str | None = None


class EvidenceLedger:
    def __init__(self) -> None:
        self._records: dict[UUID, EvidenceRecord] = {}
        self._ref_registry: dict[str, UUID] = {}
        self._text: dict[UUID, str] = {}
        self._next_ref = 1
        self._lookup_chunks: dict[int, list[UUID]] = {}
        self._lookup_count = 0
        self._protected: set[UUID] = set()

    def start_lookup(self) -> int:
        lookup_id = self._lookup_count
        self._lookup_count += 1
        self._lookup_chunks[lookup_id] = []
        return lookup_id

    def admit(self, lookup_id: int, iteration: int, chunks: Sequence[RetrievedChunk]) -> int:
        """Register the chunks one search returned. Returns the count newly admitted.

        A chunk independently surfaced by multiple searches keeps every lookup that
        found it (`seen_in_lookups`), fixing P1-5: `apply_cap` below can then keep a
        chunk any lookup ranked top-N, not only the lookup that admitted it first.
        """
        new_count = 0
        for chunk in chunks:
            record = self._records.get(chunk.chunk_id)
            if record is None:
                self._records[chunk.chunk_id] = EvidenceRecord(
                    chunk=chunk,
                    first_seen_iteration=iteration,
                    seen_in_lookups={lookup_id},
                    best_score=chunk.score or 0.0,
                )
                new_count += 1
            else:
                record.seen_in_lookups.add(lookup_id)
                record.best_score = max(record.best_score, chunk.score or 0.0)
            # Every chunk this lookup returned goes in its rank-ordered list, even one
            # already admitted by an earlier lookup — apply_cap below reads this list to
            # keep a chunk any lookup ranked top-N, not only the lookup that saw it first.
            self._lookup_chunks[lookup_id].append(chunk.chunk_id)
        return new_count

    def assign_labels(
        self,
        chunks: Sequence[RetrievedChunk],
        payloads: dict[UUID, ChunkPromptPayload],
    ) -> RAGContext:
        """Assemble one tool result's RAGContext, numbering S-labels globally across the
        request so labels never restart at S1 between searches.

        A chunk already labelled — by an earlier search this run, or seeded from a prior
        turn via `from_carryover` — keeps its one stable label and is not re-rendered:
        re-surfacing updates provenance in `admit`, never mints a second S-label. This is
        the dedup step 11's carried-evidence seeding depends on.
        """
        fresh = [c for c in chunks if c.chunk_id not in self._text]
        ctx, _ = assemble_rag_context(fresh, payloads, assume_unique=True, ref_start=self._next_ref)
        for item in ctx.items:
            self._ref_registry[item.ref_id] = item.chunk_id
            self._text[item.chunk_id] = item.prompt_text
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

    def labels_for(self, chunk_ids: Iterable[str]) -> list[str]:
        """Reverse of `resolve_refs`: chunk-UUID strings back to the S-labels the model
        was shown. Unlabelled or unknown chunks are skipped.

        Findings store resolved UUIDs, but anything rendered back into the model's view
        must speak in labels — that is the handle it can cite (P0-2).
        """
        labels: list[str] = []
        for cid_str in chunk_ids:
            try:
                record = self._records.get(UUID(cid_str))
            except ValueError:
                continue
            if record is not None and record.ref_id is not None:
                labels.append(record.ref_id)
        return labels

    def text_for(self, chunk_id: UUID) -> str | None:
        """Chunk prompt text, held in-memory since search time (P3-21)."""
        return self._text.get(chunk_id)

    def protect(self, chunk_ids: Iterable[str]) -> None:
        """Mark chunks as never-evict, additive-only. Cited by any finalizer attempt
        (accepted or rejected) this request — see the loop-internal eviction guard note
        in the target-architecture doc."""
        for cid_str in chunk_ids:
            with contextlib.suppress(ValueError):
                self._protected.add(UUID(cid_str))

    def apply_cap(self, max_per_lookup: int) -> None:
        """Post-loop, once: trim each lookup to its top-N chunks (reranker order), but
        never evict a chunk any lookup ranked top-N (P1-5) or one `protect`ed.

        This is the final synthesis-selection safety net, not the mid-loop token
        control — the loop renders top-N per search directly at transcript entry
        (P2), independently of this. `admit` above always sees the full, uncapped
        result so this cap's any-lookup-top-N accounting stays correct regardless
        of what got rendered mid-loop.
        """
        if self._lookup_count <= 1:
            return
        keep_ids: set[UUID] = set(self._protected)
        for cids in self._lookup_chunks.values():
            keep_ids.update(cids[:max_per_lookup])
        if len(keep_ids) < len(self._records):
            self._records = {cid: r for cid, r in self._records.items() if cid in keep_ids}

    def ordered(self) -> list[RetrievedChunk]:
        return sorted(
            (r.chunk for r in self._records.values()),
            key=lambda c: (c.turn_index, -(c.score or 0)),
        )

    @property
    def registry(self) -> dict[UUID, RetrievedChunk]:
        return {cid: r.chunk for cid, r in self._records.items()}

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, chunk_id: UUID) -> bool:
        return chunk_id in self._records
