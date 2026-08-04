"""Schemas for the RAG retrieval pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from pydantic import BaseModel

RetrievalSource = Literal["vector", "keyword", "hybrid"]

REF_PLACEHOLDER = "__REF__"  # safer than str.format() for arbitrary chunk text
SOURCE_REF_PREFIX = "S"


@dataclass
class RetrievedChunk:
    """Chunk returned from vector, keyword, or hybrid retrieval."""

    chunk_id: UUID
    document_id: UUID
    score: float
    chunk_index: int
    page_start: int | None
    page_end: int | None
    heading_trail: list[str]
    source: RetrievalSource
    chunk_type: str | None = None
    vector_rank: int | None = None
    vector_score: float | None = None
    keyword_rank: int | None = None
    keyword_score: float | None = None
    turn_index: int = 0  # agent loop turn that produced this chunk; 0 for non-agent paths


@dataclass(slots=True, frozen=True)
class ChunkPromptPayload:
    """Prompt-ready chunk for context assembly."""

    chunk_id: UUID
    document_id: UUID
    document_name: str
    page_numbers: tuple[int, ...]
    heading_trail: tuple[str, ...]  # e.g. ("Starvest plc Report...", "CONTENTS")
    prompt_text: str  # e.g. "[__REF__ | Doc | p.42 | Section]\n<enriched_text>"
    snippet: str | None = None
    provenance: ChunkProvenance | None = None


@dataclass(slots=True, frozen=True)
class Citation:
    ref_id: str
    ref_index: int
    chunk_id: UUID
    document_id: UUID
    document_name: str
    filename: str | None
    page_numbers: tuple[int, ...]
    heading_path: tuple[str, ...]
    snippet: str | None = None


@dataclass(slots=True, frozen=True)
class BoundingBox:
    left: float
    top: float
    right: float
    bottom: float
    coord_origin: str


@dataclass(slots=True, frozen=True)
class ProvenanceItem:
    page_no: int
    label: str
    self_ref: str | None
    charspan: tuple[int, int] | None
    bbox: BoundingBox | None


@dataclass(slots=True, frozen=True)
class ChunkProvenance:
    filename: str | None
    mimetype: str | None
    binary_hash: int | None
    page_span: tuple[int, int] | None
    doc_item_refs: tuple[str, ...]
    items: tuple[ProvenanceItem, ...]


@dataclass(slots=True, frozen=True)
class ContextItem:
    ref_id: str
    chunk_id: UUID
    score: float
    prompt_text: str
    citation: Citation
    provenance: ChunkProvenance | None = None


@dataclass(slots=True, frozen=True)
class RAGContext:
    formatted_context: str
    items: tuple[ContextItem, ...]
    chunk_count: int

    @property
    def citations(self) -> tuple[Citation, ...]:
        return tuple(item.citation for item in self.items)

    @property
    def retrieval_scores(self) -> tuple[float, ...]:
        return tuple(item.score for item in self.items)

    @property
    def chunk_to_ref(self) -> dict[UUID, str]:
        return {item.chunk_id: item.ref_id for item in self.items}

    def ref_for(self, chunk_id: UUID) -> str | None:
        return self.chunk_to_ref.get(chunk_id)


# ---------------------------------------------------------------------------
# Answer-layer citation models (streaming parser output)
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class AnswerCitationSpan:
    """A span of text in the clean (visible) answer that is supported by sources."""

    start: int  # char offset in clean text (inclusive)
    end: int  # char offset in clean text (exclusive)
    ref_ids: tuple[str, ...]  # e.g. ("S1", "S4")


@dataclass
class ParserOutput:
    """Output from feeding a chunk to the citation parser."""

    visible_text: str
    completed_spans: list[AnswerCitationSpan]


class RetrievalHit(BaseModel):
    """Compact hit for trace persistence — IDs and scores only, no text."""

    id: str
    score: float | None = None
    vector_score: float | None = None
    keyword_score: float | None = None


class DroppedChunk(BaseModel):
    chunk_id: str
    matched_rules: list[str]
    score: int


class FlaggedChunk(BaseModel):
    chunk_id: str
    matched_rules: list[str]
    score: int


class RetrievalTrace(BaseModel):
    """Per-request retrieval trace stored in message.trace['retrieval']."""

    qdrant: list[RetrievalHit] = []
    opensearch: list[RetrievalHit] = []
    fused: list[RetrievalHit] = []
    reranked: list[RetrievalHit] = []
    # Multi-pass mode: one entry per sub-query pass; single-pass fields above are absent.
    sub_passes: list[dict] | None = None
    dropped_chunks: list[DroppedChunk] = []
    flagged_chunks: list[FlaggedChunk] = []
