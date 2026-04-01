from __future__ import annotations

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from src.repository.document_repository import DocumentRepository
from src.schemas.query_router import ExtractedEntity, TimeRef
from src.utils.config import get_router_config


async def resolve_entities_to_doc_ids(
    session: AsyncSession,
    user_id: UUID,
    entities: list[ExtractedEntity],
    time_refs: list[TimeRef],
    *,
    constrain_to: list[UUID] | None = None,
    threshold: float | None = None,
    max_candidates: int | None = None,
) -> tuple[list[UUID], list[ExtractedEntity]]:
    """Map extracted entities to document IDs via pg_trgm fuzzy matching.

    Returns (matched_doc_ids, unresolved_entities).
    """
    if not entities:
        return [], []

    cfg = get_router_config()
    if threshold is None:
        threshold = cfg["entity_similarity_threshold"]
    if max_candidates is None:
        max_candidates = int(cfg["entity_max_candidates"])

    repo = DocumentRepository(session)
    resolved_years = [tr.year for tr in time_refs if tr.year is not None] or None
    matched: set[UUID] = set()
    unresolved: list[ExtractedEntity] = []

    for entity in entities:
        doc_ids = await repo.find_by_company_similarity(
            user_id,
            entity.name,
            threshold=float(threshold),
            years=resolved_years,
            constrain_to=constrain_to,
            limit=int(max_candidates),
        )
        if doc_ids:
            matched.update(doc_ids)
        else:
            unresolved.append(entity)

    return list(matched), unresolved
