from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel


class ExtractedEntity(BaseModel):
    name: str
    entity_type: str  # "company" | "person" | "product" | "unknown"
    raw_span: str  # verbatim substring from query


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
    entities: list[ExtractedEntity] = []
    user_intent: str
    reasoning: str
    query_shape: Literal["extraction", "comparison", "analytical"] | None = None
    requested_currency: str | None = (
        None  # ISO code extracted from query ("...in USD"); None if not stated
    )


class EntityManifestItem(BaseModel):
    entity_name: str
    doc_summaries: list[dict]  # [{doc_id, name, year}]


class DocumentScopeResult(BaseModel):
    doc_ids: list[UUID] | None  # None = no pre-filter (search all user docs)
    source: Literal["explicit", "filtered", "entity_resolved", "all"]
    per_entity_doc_ids: dict[str, list[UUID]] | None = None  # keyed by ExtractedEntity.name
    entity_manifest: list[EntityManifestItem] | None = None
