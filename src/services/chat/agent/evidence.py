"""EvidenceLedger — the agent's durable, chunk-level record for one run.

Holds the run's chunk/ref bookkeeping with a single invariant: every S-label the transcript
has shown resolves through `resolve_refs`. The transcript is append-only, so every label
also stays readable on screen; each chunk is rendered once per run.

Single writer: `admit`/`assign_labels` are called only on the loop task, only in the
post-`gather` reduce. Search handlers are pure and never receive a ledger reference, so
`next_ref` label allocation stays deterministic under concurrent search.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from dataclasses import replace as dc_replace
from uuid import UUID

from src.schemas.retrieval import (
    REF_PLACEHOLDER,
    ChunkPromptPayload,
    RAGContext,
    RetrievedChunk,
)
from src.services.retrieval.context_assembler import assemble_rag_context
from src.utils.config import get_agent_shown_heading_chars


@dataclass
class EvidenceRecord:
    chunk: RetrievedChunk
    # Set once the chunk is labelled, i.e. rendered into the transcript.
    ref_id: str | None = None
    # Which `assign_labels` call labelled it (one per search, in run order), and its
    # position in that search's rendered result. Meaningful only once `ref_id` is set.
    search: int = 0
    rank: int = 0


class EvidenceLedger:
    def __init__(self) -> None:
        self._records: dict[UUID, EvidenceRecord] = {}
        # Reverse index of `EvidenceRecord.ref_id`, for label lookups.
        self._ref_registry: dict[str, UUID] = {}
        # Sanitized payloads as rendered, so synthesis re-assembles without a second
        # DB hydration or a second injection scan over the same text.
        self._payloads: dict[UUID, ChunkPromptPayload] = {}
        self._next_ref = 1
        self._searches = 0

    def admit(self, chunks: Sequence[RetrievedChunk]) -> int:
        """Register the chunks one search returned. Returns the count newly admitted."""
        new_count = 0
        for chunk in chunks:
            if chunk.chunk_id not in self._records:
                self._records[chunk.chunk_id] = EvidenceRecord(chunk=chunk)
                new_count += 1
        return new_count

    def _ref_for(self, chunk_id: UUID) -> str | None:
        record = self._records.get(chunk_id)
        return record.ref_id if record is not None else None

    def shown_before(self, chunks: Sequence[RetrievedChunk]) -> list[str]:
        """`Sn heading` for each of `chunks` an earlier search already rendered, in the
        given order. Call before `assign_labels`, which labels the rest."""
        max_chars = get_agent_shown_heading_chars()
        out: list[str] = []
        for chunk in chunks:
            ref_id = self._ref_for(chunk.chunk_id)
            if ref_id is None:
                continue
            heading = chunk.heading_trail[-1] if chunk.heading_trail else ""
            if len(heading) > max_chars:
                heading = heading[:max_chars].rstrip() + "…"
            out.append(f"{ref_id} {heading}".rstrip())
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
        the transcript is append-only, so it is still on screen. Re-surfacing never mints a
        second label.
        """
        fresh = {c.chunk_id: c for c in chunks if self._ref_for(c.chunk_id) is None}
        ctx, _ = assemble_rag_context(
            list(fresh.values()), payloads, assume_unique=True, ref_start=self._next_ref
        )
        search = self._searches
        self._searches += 1
        for rank, item in enumerate(ctx.items):
            # A labelled chunk is always admitted, even if the caller skipped `admit`.
            record = self._records.setdefault(item.chunk_id, EvidenceRecord(fresh[item.chunk_id]))
            record.ref_id = item.ref_id
            record.search = search
            record.rank = rank
            self._ref_registry[item.ref_id] = item.chunk_id
            payload = payloads.get(item.chunk_id)
            if payload is not None:
                # Store the post-scan text: synthesis re-assembly then re-runs the
                # injection scan over already-sanitized text, a provable no-op.
                self._payloads[item.chunk_id] = dc_replace(
                    payload, prompt_text=item.prompt_text.replace(item.ref_id, REF_PLACEHOLDER, 1)
                )
        self._next_ref += len(ctx.items)
        return ctx

    def resolve_refs(self, refs: list[str] | None) -> tuple[list[str], list[str]]:
        """Map agent-reported S-labels (or already-UUID refs) to chunk-UUID strings.

        Returns (resolved, unresolved). Resolved refs are canonical `str(UUID)`, so callers
        can key on them interchangeably with a parsed chunk id.
        """
        resolved: list[str] = []
        unresolved: list[str] = []
        for ref in refs or []:
            candidate = ref.strip()
            try:
                chunk_id: UUID | None = UUID(candidate)
            except ValueError:
                chunk_id = self._ref_registry.get(candidate.upper())
            if chunk_id is not None:
                resolved.append(str(chunk_id))
            else:
                unresolved.append(candidate)
        return resolved, unresolved

    def payloads_for(self, chunk_ids: Iterable[UUID]) -> dict[UUID, ChunkPromptPayload]:
        """Sanitized payloads cached at render time — no second hydration. Every
        chunk synthesis can select was rendered, so a cached payload always exists."""
        return {cid: self._payloads[cid] for cid in chunk_ids if cid in self._payloads}

    def labelled_chunks(self) -> list[RetrievedChunk]:
        """Every chunk the model was shown, highest score first, globally — the only
        defensible synthesis candidates, and exactly the chunks with a cached payload.
        The score is the one from the search that first returned the chunk.
        `assemble_rag_context` assigns S-labels in input order, so this order is what makes
        synthesis's S1 the best chunk of the whole run."""
        return sorted(
            (r.chunk for r in self._records.values() if r.ref_id is not None),
            key=lambda c: -(c.score or 0),
        )

    def fallback_chunks(self, limit: int) -> list[RetrievedChunk]:
        """Up to `limit` shown chunks, round-robin across the searches that labelled them:
        every search's first chunk in run order, then every search's second, and so on.

        A search is one entity and one sub-question, so every entity and aspect is
        represented before any gets a second chunk. Only ranks within one search are
        compared: scores from different searches come from different queries, and a search
        whose reranker fell open carries fusion scores on another scale entirely.
        """
        labelled = [r for r in self._records.values() if r.ref_id is not None]
        labelled.sort(key=lambda r: (r.rank, r.search))
        return [r.chunk for r in labelled[:limit]]

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, chunk_id: UUID) -> bool:
        return chunk_id in self._records
