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
    chunk_id: UUID
    document_id: UUID
    document_name: str
    page_numbers: tuple[int, ...]
    section_title: str | None

    # Prompt-ready block with a placeholder for citation ID.
    # Example:
    #   "[__REF__ | Apple 10-K 2024 | p.42 | Risk Factors]\n<enriched_text>"
    prompt_block_template: str

    # Token count of prompt_block_template measured with a fixed-width ref
    # like "C000", so runtime replacement stays effectively stable.
    prompt_token_count: int

    # Optional short snippet for UI citation drawers / collapsible source list.
    # Not needed by the model, so keep it optional.
    snippet: str | None = None


@dataclass(slots=True, frozen=True)
class Citation:
    ref_id: str  # "C1", "C2", ...
    ref_index: int
    chunk_id: UUID
    document_id: UUID
    document_name: str
    page_numbers: tuple[int, ...]
    section_title: str | None
    snippet: str | None = None


@dataclass(slots=True, frozen=True)
class ContextItem:
    ref_id: str
    chunk_id: UUID
    score: float
    token_count: int
    prompt_text: str
    citation: Citation


@dataclass(slots=True, frozen=True)
class RAGContext:
    formatted_context: str
    items: tuple[ContextItem, ...]
    chunk_count: int
    token_count: int

    @property
    def citations(self) -> tuple[Citation, ...]:
        return tuple(item.citation for item in self.items)

    @property
    def retrieval_scores(self) -> tuple[float, ...]:
        return tuple(item.score for item in self.items)
