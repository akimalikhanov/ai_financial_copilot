from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel


class ExtractedEntity(BaseModel):
    name: str
    entity_type: str  # "company" | "person" | "product" | "unknown"
    raw_span: str  # verbatim substring from query


class TimeRef(BaseModel):
    raw: str
    year: int | None = None
    period: str | None = None  # "annual" | "H1" | "Q3" etc.


class ScopeFilters(BaseModel):
    company: list[str] = []
    year: list[int] = []
    type: list[str] = []


class ChatScope(BaseModel):
    mode: Literal["allDocs", "filteredByMetadata", "selectedDocs", "thisDoc"]
    doc_ids: list[UUID] = []
    filters: ScopeFilters = ScopeFilters()


class RouterInput(BaseModel):
    query: str
    scope: ChatScope | None = None
    conversation_history: list[dict] = []  # last 3 pairs, assistant truncated to 150 tokens


class RouterOutput(BaseModel):
    route: Literal["direct_answer", "retrieval", "out_of_scope"]
    route_confidence: float
    entities: list[ExtractedEntity] = []
    time_references: list[TimeRef] = []
    doc_type_hints: list[str] = []
    user_intent: str
    needs_decomposition: bool
    reasoning: str


class DocumentScopeResult(BaseModel):
    doc_ids: list[UUID] | None  # None = no pre-filter (search all user docs)
    source: Literal["explicit", "filtered", "entity_resolved", "all"]
    unresolved_entities: list[ExtractedEntity] = []  # Entities with no matching docs
