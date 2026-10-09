from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Literal
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from src.observability.langfuse import span as lf_span
from src.observability.trace_payload import cap_list
from src.repository.conversation_repository import ConversationRepository
from src.repository.document_repository import DocumentRepository
from src.schemas.query_router import CompanyCandidate, ExtractedEntity
from src.services.llm_router import LLMRouter
from src.services.router.company_name import normalize_company
from src.services.router.disambiguator import Disambiguation, disambiguate
from src.utils.config import get_router_config

logger = logging.getLogger(__name__)


class EntityResolution(Disambiguation):
    """How an entity resolved. Unless the decision is "none", the first candidate is the pick."""

    method: Literal["binding", "fast_path", "llm", "fallback", "no_candidates"]
    # From a binding: the user's answer when the company was outside the UI scope.
    outside_scope: Literal["include", "exclude"] | None = None


def _keys(entity: ExtractedEntity) -> list[str]:
    """Normalized lookup strings: the verbatim span and the router's expansion."""
    return [k for k in dict.fromkeys(map(normalize_company, (entity.raw_span, entity.name))) if k]


def _merge(lists: Iterable[list[CompanyCandidate]]) -> list[CompanyCandidate]:
    """One candidate per company, at its best score, best first."""
    best: dict[str, CompanyCandidate] = {}
    for c in (c for cs in lists for c in cs):
        if c.company_norm not in best or c.score > best[c.company_norm].score:
            best[c.company_norm] = c
    return sorted(best.values(), key=lambda c: c.score, reverse=True)


def _from_llm(d: Disambiguation, own: list[CompanyCandidate]) -> EntityResolution:
    """The LLM's answer, kept to the entity's own trigram candidates.

    The pool merges every pending entity's candidates, so the model can pick a company
    that matched only another entity's keys. Such a pick matched none of this entity's
    lookup strings: a guess, not a match. With nothing left the entity is "none".
    """
    norms = {c.company_norm for c in own}
    kept = [c for c in d.candidates if c.company_norm in norms]
    dropped = [c.display_name for c in d.candidates if c.company_norm not in norms]
    if dropped:
        logger.warning("entity_disambiguator_pick_outside_candidates", extra={"dropped": dropped})
    if not kept:
        return EntityResolution(decision="none", candidates=[], method="llm")
    return EntityResolution(decision=d.decision, candidates=kept, method="llm")


def _fallback(trigram: list[CompanyCandidate]) -> EntityResolution:
    """No usable LLM answer: the trigram candidates, best first, as an ambiguous result."""
    if not trigram:
        return EntityResolution(decision="none", candidates=[], method="fallback")
    return EntityResolution(decision="ambiguous", candidates=trigram, method="fallback")


async def resolve_entities(
    session: AsyncSession,
    user_id: UUID,
    query: str,
    entities: list[ExtractedEntity],
    *,
    llm_router: LLMRouter | None = None,
    parent_request_id: UUID | None = None,
    conversation_id: UUID | None = None,
) -> list[EntityResolution]:
    """Resolve each entity against all of the user's companies, in input order.

    Order: the conversation's binding from an earlier clarification, then a single exact
    match on a lookup string, then one LLM disambiguator call for the rest. The LLM picks
    only among the pg_trgm matches of the entity's lookup strings; an entity with none is
    "none" without a call. A failed call falls back to the trigram candidates.
    """
    if not entities:
        return []
    cfg = get_router_config()
    repo = DocumentRepository(session)
    keys = [_keys(e) for e in entities]
    bindings = (
        await ConversationRepository(session).get_entity_bindings(conversation_id)
        if conversation_id is not None
        else {}
    )
    bound = [next((bindings[k] for k in ks if k in bindings), None) for ks in keys]
    results: list[EntityResolution | None] = [None] * len(entities)

    with lf_span(
        "entity_candidates",
        as_type="retriever",
        input={e.name: ks for e, ks in zip(entities, keys, strict=True)},
    ) as obs:
        # A bound company is looked up by its own company_norm, which matches exactly.
        found = await repo.find_company_candidates(
            user_id,
            [k for ks in keys for k in ks] + [b["company_norm"] for b in bound if b],
            limit=int(cfg["entity_max_candidates"]),
            sim_threshold=float(cfg["entity_candidate_sim_threshold"]),
            word_threshold=float(cfg["entity_candidate_word_threshold"]),
        )
        trigram = [_merge(found[k] for k in ks) for ks in keys]

        pending: list[int] = []
        for i, ks in enumerate(keys):
            b = bound[i]
            company = (
                next(
                    (c for c in found[b["company_norm"]] if c.company_norm == b["company_norm"]),
                    None,
                )
                if b
                else None
            )
            exact = [c for c in trigram[i] if c.company_norm in ks]
            if b and company:
                results[i] = EntityResolution(
                    decision="resolved",
                    candidates=[company],
                    method="binding",
                    outside_scope=b.get("outside_scope"),
                )
            elif len(exact) == 1:
                results[i] = EntityResolution(
                    decision="resolved", candidates=exact, method="fast_path"
                )
            elif trigram[i]:
                pending.append(i)
            else:
                results[i] = EntityResolution(
                    decision="none", candidates=[], method="no_candidates"
                )

        pool = _merge(trigram[i] for i in pending)

        if obs:
            obs.update(
                output={
                    "candidates": {
                        e.name: cap_list([f"{c.display_name} ({c.score:.2f})" for c in cs])
                        for e, cs in zip(entities, trigram, strict=True)
                    },
                    "decided": {entities[i].name: r.method for i, r in enumerate(results) if r},
                    "llm_pool_size": len(pool),
                }
            )

    if pending:
        decisions = await disambiguate(
            query,
            [entities[i] for i in pending],
            pool,
            llm_router=llm_router,
            session=session,
            user_id=user_id,
            parent_request_id=parent_request_id,
            conversation_id=conversation_id,
        )
        for n, i in enumerate(pending):
            if decisions is None:
                results[i] = _fallback(trigram[i])
            else:
                results[i] = _from_llm(decisions[n], trigram[i])

    return [r for r in results if r is not None]
