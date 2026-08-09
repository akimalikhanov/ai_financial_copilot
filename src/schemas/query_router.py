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
    # Full loaded tail; the router caps it to ROUTER_HISTORY_TURNS pairs and truncates
    # assistant turns to 150 tokens when building its prompt.
    conversation_history: list[dict] = []
    # Prior turn's findings block: lets the router tell a follow-up answerable from
    # already-retrieved data from one that needs a new value out of the corpus.
    prior_findings_block: str | None = None


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
