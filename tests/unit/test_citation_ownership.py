"""Unit tests for citation ownership consolidation (docs/stages/agentic_state_refactor_v2.md step 4).

`RAGContext` is the sole owner of its chunk<->label mapping; `_map_refs` reads it instead of
callers reconstructing a `chunk_id_to_ref` dict by hand.
"""

from __future__ import annotations

from uuid import uuid4

from src.observability.metrics import CITATION_REFS_DROPPED
from src.schemas.retrieval import ChunkPromptPayload, RetrievedChunk
from src.services.chat.agent.processor import _map_refs
from src.services.retrieval.context_assembler import assemble_rag_context


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        score=1.0,
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
        prompt_text="[header]\nSome chunk text.",
    )


class TestRefFor:
    def test_returns_label_for_present_chunk(self) -> None:
        chunk = _chunk()
        ctx, _ = assemble_rag_context([chunk], {chunk.chunk_id: _payload(chunk)})

        assert ctx.ref_for(chunk.chunk_id) == "S1"

    def test_returns_none_for_absent_chunk(self) -> None:
        chunk = _chunk()
        ctx, _ = assemble_rag_context([chunk], {chunk.chunk_id: _payload(chunk)})

        assert ctx.ref_for(uuid4()) is None


class TestMapRefs:
    def test_maps_present_uuid_and_drops_absent(self) -> None:
        present, absent = _chunk(), _chunk()
        ctx, _ = assemble_rag_context([present], {present.chunk_id: _payload(present)})
        dropped_before = CITATION_REFS_DROPPED._value.get()

        result = _map_refs([str(present.chunk_id), str(absent.chunk_id)], ctx)

        assert result == "S1"
        assert CITATION_REFS_DROPPED._value.get() == dropped_before + 1

    def test_all_dropped_renders_em_dash(self) -> None:
        chunk = _chunk()
        ctx, _ = assemble_rag_context([chunk], {chunk.chunk_id: _payload(chunk)})

        assert _map_refs([str(uuid4())], ctx) == "—"
