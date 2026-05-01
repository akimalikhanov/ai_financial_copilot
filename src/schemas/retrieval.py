"""Schemas for the RAG retrieval pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
from uuid import UUID

from pydantic import BaseModel

RetrievalSource = Literal["vector", "keyword", "hybrid"]

Route = Literal["direct_answer", "retrieve", "out_of_scope"]


class RouterOutput(BaseModel):
    """Schema for LLM router response. LLM must output valid JSON matching this structure."""

    route: Route
    user_intent: str
    reason: str | None = None


@dataclass(slots=True, frozen=True)
class ProcessedQuery:
    """Result of query preprocessing and routing."""

    normalized_text: str
    route: Route
    user_intent: str
    reason: str | None = None


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
class DisplayLabelMap:
    """Maps source ref_ids to presentation-layer display labels (C1, C2, ...).

    Labels are assigned sequentially by first appearance in the answer text,
    so the display order is independent of retrieval/reranker order.
    """

    _source_to_label: dict[str, str] = field(default_factory=dict)
    _next_index: int = field(default=1)

    def get_or_assign(self, ref_id: str) -> str:
        """Get existing label or assign next sequential one."""
        if ref_id not in self._source_to_label:
            self._source_to_label[ref_id] = f"C{self._next_index}"
            self._next_index += 1
        return self._source_to_label[ref_id]

    def get_labels_for_refs(self, ref_ids: tuple[str, ...]) -> tuple[str, ...]:
        """Get or assign labels for multiple refs, preserving order."""
        return tuple(self.get_or_assign(r) for r in ref_ids)

    @property
    def mapping(self) -> dict[str, str]:
        """Return a copy of the current source-to-display mapping."""
        return dict(self._source_to_label)


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
