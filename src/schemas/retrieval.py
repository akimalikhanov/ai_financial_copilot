"""Schemas for the RAG retrieval pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from uuid import UUID

RetrievalSource = Literal["vector", "keyword", "hybrid"]

REF_PLACEHOLDER = "__REF__"  # safer than str.format() for arbitrary chunk text


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
