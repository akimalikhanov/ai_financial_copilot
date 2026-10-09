from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel

from src.schemas.chat import Turn


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
    # Every prior turn: the router indexes all of them and shows the recent ones in full.
    prior_turns: list[Turn] = []
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
    # What the question called the company when that isn't its name ("PFH" for a
    # company the user picked on a clarification card).
    mentioned_as: list[str] = []


class CompanyCandidate(BaseModel):
    """One company (a distinct `company_norm`) with its ready documents."""

    company_norm: str
    display_name: str
    doc_ids: list[UUID]
    years: list[int]
    titles: list[str]
    # Best of similarity / strict_word_similarity against the lookup key; 0 when listed
    # from the catalogue rather than matched.
    score: float = 0.0


class EntityClarification(BaseModel):
    """An entity the clarification card asks about. With clarification off, the run goes
    ahead: an ambiguous entity uses its first candidate, the others count as not found."""

    entity: str  # ExtractedEntity.name
    raw_span: str
    outcome: Literal["ambiguous", "none", "outside_scope"]
    candidates: list[CompanyCandidate] = []


ScopeSource = Literal["explicit", "filtered", "entity_resolved", "unresolved", "all"]


class DocumentScopeResult(BaseModel):
    # None = no pre-filter. resolve_scope always returns a list; None stays the retrievers'
    # contract for callers that build a result themselves.
    doc_ids: list[UUID] | None
    # Without entities, the UI scope the documents came from (explicit selection, metadata
    # filter or all). With entities, whether any of them resolved.
    source: ScopeSource
    # Covered company display name → its documents in the UI scope; an unresolved entity
    # is listed under its router name with [].
    per_entity_doc_ids: dict[str, list[UUID]] | None = None
    # Entity names that matched no document in the UI scope.
    unresolved_entities: list[str] = []
    entity_manifest: list[EntityManifestItem] | None = None
    clarifications: list[EntityClarification] = []
    # Companies covered, set only when above SCOPE_MAX_COMPANIES: the run stops before the agent.
    too_broad_count: int | None = None

    def mentions(self) -> dict[str, str]:
        """Covered company → how the question named it, quoted, where that isn't its name."""
        return {
            item.entity_name: ", ".join(f'"{m}"' for m in item.mentioned_as)
            for item in self.entity_manifest or []
            if item.mentioned_as
        }
