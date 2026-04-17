from __future__ import annotations

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from src.repository.document_repository import DocumentRepository
from src.schemas.query_router import ExtractedEntity
from src.utils.config import get_router_config


async def _resolve_all_entities(
    session: AsyncSession,
    user_id: UUID,
    entities: list[ExtractedEntity],
    *,
    constrain_to: list[UUID] | None = None,
    threshold: float,
    max_candidates: int,
) -> dict[str, list[UUID]]:
    """Resolve entity lookups sequentially on a shared session.

    AsyncSession does not support concurrent operations, so lookups run
    in sequence. Returns {entity.name: [doc_ids]}.
    """
    repo = DocumentRepository(session)
    result: dict[str, list[UUID]] = {}
    for entity in entities:
        result[entity.name] = await repo.find_by_company_similarity(
            user_id,
            entity.name,
            threshold=threshold,
            constrain_to=constrain_to,
            limit=max_candidates,
        )
    return result


async def resolve_entities_to_doc_ids(
    session: AsyncSession,
    user_id: UUID,
    entities: list[ExtractedEntity],
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
        threshold = float(cfg["entity_similarity_threshold"])
    if max_candidates is None:
        max_candidates = int(cfg["entity_max_candidates"])

    per_entity = await _resolve_all_entities(
        session,
        user_id,
        entities,
        constrain_to=constrain_to,
        threshold=threshold,
        max_candidates=max_candidates,
    )

    matched: set[UUID] = set()
    unresolved: list[ExtractedEntity] = []
    for entity in entities:
        doc_ids = per_entity[entity.name]
        if doc_ids:
            matched.update(doc_ids)
        else:
            unresolved.append(entity)

    return list(matched), unresolved


async def resolve_entities_per_entity(
    session: AsyncSession,
    user_id: UUID,
    entities: list[ExtractedEntity],
    *,
    constrain_to: list[UUID] | None = None,
    threshold: float | None = None,
    max_candidates: int | None = None,
) -> dict[str, list[UUID]]:
    """Returns {entity.name: [doc_ids]}, including empty lists for unresolved."""
    if not entities:
        return {}

    cfg = get_router_config()
    if threshold is None:
        threshold = float(cfg["entity_similarity_threshold"])
    if max_candidates is None:
        max_candidates = int(cfg["entity_max_candidates"])

    return await _resolve_all_entities(
        session,
        user_id,
        entities,
        constrain_to=constrain_to,
        threshold=threshold,
        max_candidates=max_candidates,
    )
