from __future__ import annotations

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from src.repository.document_repository import DocumentRepository
from src.schemas.query_router import ChatScope, DocumentScopeResult, RouterOutput
from src.services.router.entity_resolver import _resolve_all_entities
from src.utils.config import get_router_config


async def _group_docs_by_company(
    session: AsyncSession,
    user_id: UUID,
    doc_ids: list[UUID],
) -> dict[str, list[UUID]] | None:
    """Group explicitly-scoped doc_ids by their company metadata.

    The agent loop identifies which company each search targets via per_entity_doc_ids.
    When the user scopes by selecting documents (rather than naming a company in the
    query), the company names live in document metadata — derive per-entity grouping
    from there so the agent knows the entity name (e.g. answering "this company").

    Docs without a company value are grouped under a single "Selected documents" bucket
    so they remain searchable. Returns None when there are no docs.
    """
    if not doc_ids:
        return None
    repo = DocumentRepository(session)
    summaries = await repo.get_scope_doc_summaries(user_id, doc_ids, limit=len(doc_ids))
    per_entity: dict[str, list[UUID]] = {}
    unnamed: list[UUID] = []
    for doc_id, company, _year in summaries:
        if company:
            per_entity.setdefault(company, []).append(doc_id)
        else:
            unnamed.append(doc_id)
    if unnamed:
        per_entity.setdefault("Selected documents", []).extend(unnamed)
    return per_entity or None


async def resolve_scope(
    session: AsyncSession,
    user_id: UUID,
    scope: ChatScope | None,
    router_output: RouterOutput,
) -> DocumentScopeResult:
    """Resolve document scope using two layers:
    Layer 1 — user-explicit scope (selectedDocs/thisDoc/filteredByMetadata).
    Layer 2 — entity narrowing (allDocs or broad filteredByMetadata).
    """
    cfg = get_router_config()
    filtered_md_thresh = int(cfg["filtered_md_thresh"])
    has_entities = bool(router_output.entities)

    # --- selectedDocs / thisDoc: use doc_ids directly, skip Layer 2 ---
    # If the user picked this mode but selected zero docs (e.g. clicked "Selected" without
    # actually choosing), fall through to entity resolution rather than searching everything.
    if scope is not None and scope.mode in ("selectedDocs", "thisDoc") and scope.doc_ids:
        return DocumentScopeResult(
            doc_ids=scope.doc_ids,
            source="explicit",
            per_entity_doc_ids=await _group_docs_by_company(session, user_id, scope.doc_ids),
        )
    # Empty selection — treat as allDocs and let entity resolution narrow it down below

    # --- filteredByMetadata: resolve filters (Layer 1), then optionally narrow (Layer 2) ---
    if scope is not None and scope.mode == "filteredByMetadata":
        repo = DocumentRepository(session)
        layer1_ids = await repo.find_by_metadata_filters(
            user_id,
            companies=scope.filters.company or None,
            years=scope.filters.year or None,
            types=scope.filters.type or None,
        )

        # Small result set — treat like selectedDocs, skip Layer 2
        if len(layer1_ids) <= filtered_md_thresh:
            return DocumentScopeResult(
                doc_ids=layer1_ids or None,
                source="filtered",
                per_entity_doc_ids=await _group_docs_by_company(session, user_id, layer1_ids),
            )

        # Large result set + entities → intersect with entity resolution
        if has_entities:
            per_entity = await _resolve_all_entities(
                session,
                user_id,
                router_output.entities,
                constrain_to=layer1_ids,
                threshold=float(cfg["entity_similarity_threshold"]),
                max_candidates=int(cfg["entity_max_candidates"]),
            )
            matched = list({doc_id for ids in per_entity.values() for doc_id in ids})
            return DocumentScopeResult(
                doc_ids=matched if matched else layer1_ids,
                source="filtered",
                per_entity_doc_ids=per_entity if per_entity else None,
            )

        # Large result set, no entities → return Layer 1, grouped by company so the
        # agent still knows the entity names of the filtered documents.
        return DocumentScopeResult(
            doc_ids=layer1_ids,
            source="filtered",
            per_entity_doc_ids=await _group_docs_by_company(session, user_id, layer1_ids),
        )

    # --- allDocs / None → Layer 2 does the heavy lifting ---
    if has_entities:
        per_entity = await _resolve_all_entities(
            session,
            user_id,
            router_output.entities,
            threshold=float(cfg["entity_similarity_threshold"]),
            max_candidates=int(cfg["entity_max_candidates"]),
        )
        matched = list({doc_id for ids in per_entity.values() for doc_id in ids})
        if matched:
            return DocumentScopeResult(
                doc_ids=matched,
                source="entity_resolved",
                per_entity_doc_ids=per_entity if per_entity else None,
            )
        return DocumentScopeResult(
            doc_ids=None,
            source="all",
        )

    # No entities, no scope → no pre-filter
    return DocumentScopeResult(doc_ids=None, source="all")
