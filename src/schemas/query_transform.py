from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel

from src.schemas.query_router import ExtractedEntity


class ScopeDocSummary(BaseModel):
    document_id: UUID
    company: str | None
    year: int | None


class SubQuery(BaseModel):
    semantic_query: str
    keyword_query: str
    focus_entity: str  # required; resolved against per_entity_doc_ids
    entity_match_quality: str = ""  # set post-parse: "exact" | "dropped"


class TransformedQuery(BaseModel):
    semantic_query: str
    keyword_query: str
    sub_queries: list[SubQuery] = []
    fallback: bool = False  # True iff LLM failed → rewrites == raw
    decomposition_overridden: bool = False  # True if guard disabled decomposition


class TransformerInput(BaseModel):
    user_query_raw: str
    conversation_history: list[dict] = []  # [{role, content}, ...]
    router_entities: list[ExtractedEntity] = []
    user_intent: str
    needs_decomposition: bool
    scope_docs: list[ScopeDocSummary] = []  # always populated (UI caps scope at 8-10 docs)
    known_entity_names: list[str] = []  # from per_entity_doc_ids keys, for prompt enumeration
