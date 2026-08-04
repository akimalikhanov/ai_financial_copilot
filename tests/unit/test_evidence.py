"""Unit tests for EvidenceLedger (docs/stages/agentic_state_refactor_v2.md step 6)."""

from __future__ import annotations

from dataclasses import replace
from uuid import UUID, uuid4

from src.schemas.retrieval import ChunkPromptPayload, RetrievedChunk
from src.services.chat.agent.evidence import EvidenceLedger


def _chunk(score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        score=score,
        chunk_index=0,
        page_start=1,
        page_end=1,
        heading_trail=[],
        source="vector",
    )


def _chunk_like(chunk: RetrievedChunk, score: float) -> RetrievedChunk:
    """Same chunk_id, different score — a re-surfaced chunk from a later search."""
    return replace(chunk, score=score)


def _payload(chunk: RetrievedChunk) -> ChunkPromptPayload:
    return ChunkPromptPayload(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        document_name="Doc.pdf",
        page_numbers=(1,),
        heading_trail=("Section",),
        prompt_text=f"[header]\nchunk {chunk.chunk_id} text.",
    )


class TestAdmit:
    def test_dedups_across_searches(self) -> None:
        ledger = EvidenceLedger()
        chunk = _chunk()
        assert ledger.admit([chunk]) == 1
        assert ledger.admit([chunk]) == 0  # already known — not newly admitted
        assert len(ledger) == 1

    def test_best_score_is_kept_across_searches(self) -> None:
        ledger = EvidenceLedger()
        chunk = _chunk(score=0.3)
        ledger.admit([chunk])
        ledger.admit([_chunk_like(chunk, score=0.8)])

        assert ledger._records[chunk.chunk_id].best_score == 0.8


class TestAssignLabels:
    def test_labels_continue_across_calls(self) -> None:
        ledger = EvidenceLedger()
        c1, c2 = _chunk(), _chunk()
        ctx1 = ledger.assign_labels([c1], {c1.chunk_id: _payload(c1)})
        ctx2 = ledger.assign_labels([c2], {c2.chunk_id: _payload(c2)})

        assert ctx1.items[0].ref_id == "S1"
        assert ctx2.items[0].ref_id == "S2"

    def test_resurfaced_chunk_keeps_one_stable_label(self) -> None:
        """Blocker 2: a chunk re-surfaced by a later search is not re-labelled or
        re-rendered — the dedup step 11's carried-evidence seeding relies on."""
        ledger = EvidenceLedger()
        c1, c2 = _chunk(), _chunk()
        payloads = {c1.chunk_id: _payload(c1), c2.chunk_id: _payload(c2)}

        ctx1 = ledger.assign_labels([c1], payloads)
        assert ctx1.items[0].ref_id == "S1"

        # A second search returns c1 again (re-surfaced) plus a fresh c2.
        ctx2 = ledger.assign_labels([c1, c2], payloads)
        # c1 is not re-rendered; only the fresh c2 gets a label, continuing at S2.
        assert [item.chunk_id for item in ctx2.items] == [c2.chunk_id]
        assert ctx2.items[0].ref_id == "S2"

        # c1 still resolves under its one original label; no second label was minted.
        resolved, _ = ledger.resolve_refs(["S1", "S2"])
        assert resolved == [str(c1.chunk_id), str(c2.chunk_id)]

    def test_resolve_refs_maps_label_to_uuid(self) -> None:
        ledger = EvidenceLedger()
        c1 = _chunk()
        ledger.assign_labels([c1], {c1.chunk_id: _payload(c1)})

        resolved, unresolved = ledger.resolve_refs(["S1", "s1", "not-a-label"])
        assert resolved == [str(c1.chunk_id), str(c1.chunk_id)]
        assert unresolved == ["not-a-label"]

    def test_resolve_refs_passes_through_uuids(self) -> None:
        ledger = EvidenceLedger()
        cid = uuid4()
        resolved, unresolved = ledger.resolve_refs([str(cid)])
        assert resolved == [str(cid)]
        assert unresolved == []


class TestRenderedChunks:
    def test_only_rendered_chunks_are_returned(self) -> None:
        """Admitted-but-never-rendered chunks are not a legitimate synthesis fallback —
        the model never saw them."""
        ledger = EvidenceLedger()
        shown, unshown = _chunk(), _chunk()
        ledger.admit([shown, unshown])
        ledger.assign_labels([shown], {shown.chunk_id: _payload(shown)})

        assert [c.chunk_id for c in ledger.rendered_chunks()] == [shown.chunk_id]

    def test_evicted_chunk_leaves_the_rendered_set(self) -> None:
        ledger = EvidenceLedger()
        c1 = _chunk()
        ledger.admit([c1])
        ctx = ledger.assign_labels([c1], {c1.chunk_id: _payload(c1)})

        ledger.mark_evicted([ctx.items[0].ref_id])
        assert ledger.rendered_chunks() == []
        # ...but it still resolves — Contract C1 is untouched by eviction.
        resolved, _ = ledger.resolve_refs([ctx.items[0].ref_id])
        assert resolved == [str(c1.chunk_id)]


class TestContractC2:
    def test_labels_deterministic_over_reduce_order(self) -> None:
        """Contract C2: same lookups reduced in the same order yield identical labels."""
        c1, c2, c3 = _chunk(), _chunk(), _chunk()
        payloads = {c.chunk_id: _payload(c) for c in (c1, c2, c3)}

        def _run() -> dict[UUID, str]:
            ledger = EvidenceLedger()
            ctx_a = ledger.assign_labels(
                [c1, c2], {c1.chunk_id: payloads[c1.chunk_id], c2.chunk_id: payloads[c2.chunk_id]}
            )
            ctx_b = ledger.assign_labels([c3], {c3.chunk_id: payloads[c3.chunk_id]})
            return {item.chunk_id: item.ref_id for item in (*ctx_a.items, *ctx_b.items)}

        first = _run()
        second = _run()
        assert first == second
