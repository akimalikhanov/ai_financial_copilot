from __future__ import annotations

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from src.observability.langfuse import span as lf_span
from src.observability.trace_payload import cap_list
from src.repository.document_repository import DocumentRepository
from src.schemas.query_router import (
    ChatScope,
    DocumentScopeResult,
    EntityClarification,
    EntityManifestItem,
    RouterOutput,
    ScopeSource,
)
from src.services.llm_router import LLMRouter
from src.services.router.entity_resolver import resolve_entities
from src.utils.config import get_scope_max_companies


def scope_outcome(router_output: RouterOutput, scope: DocumentScopeResult) -> str:
    """The request's `llm_requests.scope_outcome`, in order of precedence. `clarification`
    means the card was shown, which only the caller knows."""
    if scope.too_broad_count is not None:
        return "too_broad"
    if scope.unresolved_entities:
        return "unresolved"
    return "resolved" if router_output.entities else "no_entities"


# Documents uploaded without a company count together as one company.
NO_COMPANY = "Documents without a company"

_Doc = tuple[UUID, str | None, str | None, int | None]  # id, company_norm, company, year


async def _universe(
    repo: DocumentRepository, user_id: UUID, scope: ChatScope | None
) -> tuple[list[_Doc], ScopeSource]:
    """The documents the UI scope allows, and which kind of scope produced them. An empty
    selection counts as all documents."""
    if scope is not None and scope.mode in ("selectedDocs", "thisDoc") and scope.doc_ids:
        return await repo.get_scope_docs(user_id, scope.doc_ids), "explicit"
    if scope is not None and scope.mode == "filteredByMetadata":
        ids = await repo.find_by_metadata_filters(
            user_id,
            companies=scope.filters.company or None,
            years=scope.filters.year or None,
            types=scope.filters.type or None,
        )
        return await repo.get_scope_docs(user_id, ids), "filtered"
    return await repo.get_scope_docs(user_id), "all"


def _by_company(docs: list[_Doc]) -> dict[str, list[_Doc]]:
    """Universe documents per company display name, keyed via company_norm."""
    groups: dict[str | None, list[_Doc]] = {}
    for doc in docs:
        groups.setdefault(doc[1], []).append(doc)
    return {
        NO_COMPANY if norm is None else min(d[2] or norm for d in group): group
        for norm, group in groups.items()
    }


async def resolve_scope(
    session: AsyncSession,
    user_id: UUID,
    scope: ChatScope | None,
    router_output: RouterOutput,
    *,
    query: str = "",
    llm_router: LLMRouter | None = None,
    parent_request_id: UUID | None = None,
    conversation_id: UUID | None = None,
) -> DocumentScopeResult:
    """Scope in three steps, the same in every UI scope mode:

    1. Universe: the documents the UI scope allows.
    2. Entities: resolved against all of the user's companies, so "not in your documents"
       and "not in your current selection" stay distinguishable.
    3. Combine: each resolved entity keeps its documents inside the universe. A question
       naming no company covers every company in the universe. More than
       SCOPE_MAX_COMPANIES covered companies is too broad.

    The keyword arguments feed the entity disambiguator's LLM call and its sub-request row.
    """
    repo = DocumentRepository(session)
    with lf_span(
        "resolve_scope", input={"entities": [e.name for e in router_output.entities]}
    ) as root:
        with lf_span(
            "scope_universe",
            input=scope.model_dump(mode="json") if scope else {"mode": "allDocs"},
        ) as obs:
            docs, source = await _universe(repo, user_id, scope)
            companies = _by_company(docs)
            if obs:
                obs.update(output={"doc_count": len(docs), "company_count": len(companies)})

        covered: dict[str, list[_Doc]] = {}
        unresolved: list[str] = []
        clarifications: list[EntityClarification] = []
        if router_output.entities:
            resolutions = await resolve_entities(
                session,
                user_id,
                query,
                router_output.entities,
                llm_router=llm_router,
                parent_request_id=parent_request_id,
                conversation_id=conversation_id,
            )
            with lf_span(
                "scope_combine",
                input={
                    e.name: {
                        "decision": r.decision,
                        "method": r.method,
                        "candidates": cap_list([c.display_name for c in r.candidates]),
                    }
                    for e, r in zip(router_output.entities, resolutions, strict=True)
                },
            ) as obs:
                outcomes: dict[str, str] = {}
                for entity, res in zip(router_output.entities, resolutions, strict=True):
                    pick = res.candidates[0] if res.decision != "none" else None
                    inside = [d for d in docs if pick and d[1] == pick.company_norm]
                    if pick and not inside and res.outside_scope == "include":
                        # The user chose to search this company despite the UI scope.
                        inside = await repo.get_scope_docs(user_id, pick.doc_ids)
                    if pick and inside:
                        covered[pick.display_name] = inside
                        outcomes[entity.name] = f"{res.decision}: {pick.display_name}"
                        if res.decision == "resolved":
                            continue
                        outcome = "ambiguous"
                    else:
                        outcome = "outside_scope" if pick else "none"
                        unresolved.append(entity.name)
                        outcomes[entity.name] = outcome
                        if res.outside_scope == "exclude":
                            continue  # already answered: leave it out, don't ask again
                    # Only companies get a card; persons and products keep the plain
                    # "not found" label.
                    if entity.entity_type != "company":
                        continue
                    clarifications.append(
                        EntityClarification(
                            entity=entity.name,
                            raw_span=entity.raw_span,
                            outcome=outcome,
                            candidates=res.candidates,
                        )
                    )
                source = "entity_resolved" if covered else "unresolved"
                if obs:
                    obs.update(output={"outcomes": outcomes, "covered": list(covered)})
        else:
            covered = companies

        max_companies = get_scope_max_companies()
        if len(covered) > max_companies:
            result = DocumentScopeResult(
                doc_ids=[],
                source=source,
                unresolved_entities=unresolved,
                clarifications=clarifications,
                too_broad_count=len(covered),
            )
        else:
            per_entity = {name: [d[0] for d in group] for name, group in covered.items()}
            per_entity.update({name: [] for name in unresolved})
            result = DocumentScopeResult(
                doc_ids=[d for ids in per_entity.values() for d in ids],
                source=source,
                per_entity_doc_ids=per_entity or None,
                unresolved_entities=unresolved,
                entity_manifest=[
                    EntityManifestItem(
                        entity_name=name,
                        doc_summaries=[
                            {"doc_id": str(d[0]), "name": d[2], "year": d[3]} for d in group
                        ],
                    )
                    for name, group in covered.items()
                ]
                or None,
                clarifications=clarifications,
            )
        if root:
            root.update(
                output={
                    "source": result.source,
                    "covered": cap_list(list(covered)),
                    "too_broad": result.too_broad_count is not None,
                    "max_companies": max_companies,
                    "doc_count": len(result.doc_ids or []),
                }
            )
        return result
