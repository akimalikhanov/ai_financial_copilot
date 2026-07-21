"""Unit tests for EvidenceLedger (docs/stages/agentic_state_refactor_v2.md step 6)."""

from __future__ import annotations

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
    def test_dedups_across_lookups(self) -> None:
        ledger = EvidenceLedger()
        chunk = _chunk()
        lookup1 = ledger.start_lookup()
        assert ledger.admit(lookup1, 0, [chunk]) == 1
        lookup2 = ledger.start_lookup()
        assert ledger.admit(lookup2, 1, [chunk]) == 0  # already known — not newly admitted
        assert len(ledger) == 1

    def test_multi_lookup_provenance_tracked(self) -> None:
        ledger = EvidenceLedger()
        chunk = _chunk()
        lookup1 = ledger.start_lookup()
        ledger.admit(lookup1, 0, [chunk])
        lookup2 = ledger.start_lookup()
        ledger.admit(lookup2, 1, [chunk])

        record = ledger._records[chunk.chunk_id]
        assert record.seen_in_lookups == {lookup1, lookup2}


class TestAssignLabels:
    def test_labels_continue_across_calls(self) -> None:
        ledger = EvidenceLedger()
        c1, c2 = _chunk(), _chunk()
        ctx1 = ledger.assign_labels([c1], {c1.chunk_id: _payload(c1)})
        ctx2 = ledger.assign_labels([c2], {c2.chunk_id: _payload(c2)})

        assert ctx1.items[0].ref_id == "S1"
        assert ctx2.items[0].ref_id == "S2"

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


class TestProtect:
    def test_protect_is_additive(self) -> None:
        ledger = EvidenceLedger()
        c1, c2 = _chunk(), _chunk()
        ledger.protect([str(c1.chunk_id)])
        ledger.protect([str(c2.chunk_id)])
        assert ledger._protected == {c1.chunk_id, c2.chunk_id}

    def test_protect_ignores_malformed_ids(self) -> None:
        ledger = EvidenceLedger()
        ledger.protect(["not-a-uuid"])
        assert ledger._protected == set()


class TestApplyCap:
    def test_single_lookup_never_capped(self) -> None:
        ledger = EvidenceLedger()
        chunks = [_chunk() for _ in range(10)]
        lookup = ledger.start_lookup()
        ledger.admit(lookup, 0, chunks)
        ledger.apply_cap(max_per_lookup=2)
        assert len(ledger) == 10

    def test_keeps_any_lookup_top_n(self) -> None:
        """P1-5: a chunk ranked top-N by *any* lookup survives, not only its first lookup."""
        ledger = EvidenceLedger()
        shared, other_a, other_b = _chunk(), _chunk(), _chunk()

        lookup1 = ledger.start_lookup()
        # `shared` ranks low (3rd) in lookup1 — would be capped if only lookup1 mattered.
        ledger.admit(lookup1, 0, [other_a, other_b, shared])

        lookup2 = ledger.start_lookup()
        # `shared` ranks 1st in lookup2, along with a fresh chunk.
        fresh = _chunk()
        ledger.admit(lookup2, 1, [shared, fresh])

        ledger.apply_cap(max_per_lookup=1)

        # lookup1 top-1 = other_a, lookup2 top-1 = shared: both survive; other_b/fresh don't.
        assert set(ledger.registry.keys()) == {other_a.chunk_id, shared.chunk_id}

    def test_protected_chunk_survives_cap_regardless_of_rank(self) -> None:
        ledger = EvidenceLedger()
        top, protected_low_rank, evict = _chunk(), _chunk(), _chunk()
        lookup1 = ledger.start_lookup()
        # Without protection, max_per_lookup=1 keeps only `top`.
        ledger.admit(lookup1, 0, [top, protected_low_rank, evict])
        lookup2 = ledger.start_lookup()
        ledger.admit(lookup2, 1, [_chunk()])  # a second lookup, so capping actually runs

        ledger.protect([str(protected_low_rank.chunk_id)])
        ledger.apply_cap(max_per_lookup=1)

        assert top.chunk_id in ledger.registry
        assert protected_low_rank.chunk_id in ledger.registry
        assert evict.chunk_id not in ledger.registry


class TestOrdered:
    def test_orders_by_turn_index_then_score_desc(self) -> None:
        ledger = EvidenceLedger()
        early_low = _chunk(score=0.1)
        early_low.turn_index = 0
        early_high = _chunk(score=0.9)
        early_high.turn_index = 0
        late = _chunk(score=0.5)
        late.turn_index = 1

        lookup = ledger.start_lookup()
        ledger.admit(lookup, 0, [early_low, early_high, late])

        ordered_ids = [c.chunk_id for c in ledger.ordered()]
        assert ordered_ids == [early_high.chunk_id, early_low.chunk_id, late.chunk_id]


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
