from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel


class ScopeDocSummary(BaseModel):
    document_id: UUID
    company: str | None
    year: int | None


class TransformedQuery(BaseModel):
    semantic_query: str
    keyword_query: str
    fallback: bool = False  # True iff LLM failed → rewrites == raw


class TransformerInput(BaseModel):
    user_query_raw: str
    conversation_history: list[dict] = []  # [{role, content}, ...]
    user_intent: str
    scope_docs: list[ScopeDocSummary] = []
