"""Entity disambiguator: one LLM call that maps each entity named in a question to the
user's companies.

The resolver calls it only for entities with no single exact match. The model sees the
question, each entity's span and expanded name, and the candidate companies as short IDs
(C1…Ck) constrained by a per-request enum, so it can't name a company that isn't listed.
For each entity it returns ``resolved``, ``ambiguous`` (every plausible ID, most likely
first) or ``none``. A failed, timed-out or unparseable call returns None and the caller
falls back to the trigram candidates as an ambiguous result.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from src.observability.langfuse import describe_error
from src.observability.langfuse import mark_current as lf_mark_current
from src.observability.metrics import observe_llm_latency
from src.repository.llm_request_repository import LLMRequestRepository, stats_to_request_kwargs
from src.schemas.query_router import CompanyCandidate, ExtractedEntity
from src.services.llm_adapters.base_adapter import ChatMessage, Role
from src.services.llm_router import LLMRouter, get_router
from src.services.prompts.prompt_loader import get_prompt_loader
from src.services.prompts.prompt_renderer import get_prompt_renderer
from src.utils.config import (
    get_entity_disambiguator_model,
    get_entity_disambiguator_prompt_version,
    get_router_config,
)
from src.utils.json_schema import build_response_format

logger = logging.getLogger(__name__)

Decision = Literal["resolved", "ambiguous", "none"]


class Disambiguation(BaseModel):
    decision: Decision
    candidates: list[CompanyCandidate]  # most likely first; empty for "none"


class _Pick(BaseModel):
    entity: str
    decision: Decision
    candidates: list[str]


class _Picks(BaseModel):
    entities: list[_Pick]


def _response_format(n_entities: int, n_candidates: int) -> dict:
    """Entity and company IDs as enums, so the model can only name listed companies."""
    # make_strict doesn't descend into array items, so the item is strict by hand.
    item = {
        "type": "object",
        "required": ["entity", "decision", "candidates"],
        "additionalProperties": False,
        "properties": {
            "entity": {"type": "string", "enum": [f"E{i}" for i in range(1, n_entities + 1)]},
            "decision": {"type": "string", "enum": ["resolved", "ambiguous", "none"]},
            "candidates": {
                "type": "array",
                "items": {"type": "string", "enum": [f"C{i}" for i in range(1, n_candidates + 1)]},
            },
        },
    }
    schema = {
        "type": "object",
        "properties": {"entities": {"type": "array", "items": item}},
    }
    return build_response_format("entity_disambiguator", schema)


def _user_message(
    query: str,
    entities: list[ExtractedEntity],
    candidates: list[CompanyCandidate],
    max_titles: int,
) -> str:
    lines = [f"Question: {query}", "", "Entities:"]
    lines += [
        f'E{i}: span "{e.raw_span}", expanded name "{e.name}"' for i, e in enumerate(entities, 1)
    ]
    lines += ["", "Companies:"]
    for i, c in enumerate(candidates, 1):
        line = f"C{i}: {c.display_name} | years: {', '.join(map(str, sorted(c.years))) or '?'}"
        if c.titles[:max_titles]:
            line += f" | titles: {'; '.join(c.titles[:max_titles])}"
        lines.append(line)
    return "\n".join(lines)


def _map(
    picks: _Picks, entities: list[ExtractedEntity], candidates: list[CompanyCandidate]
) -> list[Disambiguation]:
    by_id = {f"C{i}": c for i, c in enumerate(candidates, 1)}
    by_entity = {p.entity: p for p in picks.entities}
    result: list[Disambiguation] = []
    for i in range(1, len(entities) + 1):
        pick = by_entity.get(f"E{i}")
        chosen = [by_id[c] for c in dict.fromkeys(pick.candidates) if c in by_id] if pick else []
        if pick is None or pick.decision == "none" or not chosen:
            result.append(Disambiguation(decision="none", candidates=[]))
        else:
            result.append(Disambiguation(decision=pick.decision, candidates=chosen))
    return result


async def disambiguate(
    query: str,
    entities: list[ExtractedEntity],
    candidates: list[CompanyCandidate],
    *,
    llm_router: LLMRouter | None = None,
    session: AsyncSession | None = None,
    user_id: UUID | None = None,
    parent_request_id: UUID | None = None,
    conversation_id: UUID | None = None,
) -> list[Disambiguation] | None:
    """One LLM call for all entities: a decision per entity, in input order.

    Returns None when the call fails, times out or returns unusable output; the caller
    falls back. With a session and parent request, the call is logged as an
    ``entity_disambiguator`` sub-request, failures included.
    """
    cfg = get_router_config()
    model_id = get_entity_disambiguator_model()
    max_tokens = int(cfg["disambiguator_max_tokens"])
    try:
        llm = (llm_router or get_router()).get(model_id)
        prompt = get_prompt_loader().load(
            "entity_disambiguator", get_entity_disambiguator_prompt_version()
        )
        system = get_prompt_renderer()._render_template(prompt.template, {})
    except Exception as e:
        logger.warning("entity_disambiguator_unavailable", extra={"model": model_id})
        lf_mark_current("WARNING", f"disambiguator unavailable ({describe_error(e)}); fallback")
        return None

    async def log(**kwargs: Any) -> None:
        if session is None or parent_request_id is None or conversation_id is None:
            return
        await LLMRequestRepository(session).create_subrequest(
            parent_request_id=parent_request_id,
            conversation_id=conversation_id,
            user_id=user_id,
            provider=llm.provider,
            model=model_id,
            request_type="entity_disambiguator",
            request_params={"temperature": cfg["temperature"], "max_tokens": max_tokens},
            **kwargs,
        )

    messages = [
        ChatMessage(role=Role.system, content=system),
        ChatMessage(
            role=Role.user,
            content=_user_message(
                query, entities, candidates, int(cfg["disambiguator_max_titles"])
            ),
        ),
    ]
    try:
        resp = await asyncio.wait_for(
            llm.complete(
                messages=messages,
                _lf_name="entity_disambiguator",
                temperature=cfg["temperature"],
                max_tokens=max_tokens,
                response_format=_response_format(len(entities), len(candidates)),
            ),
            timeout=float(cfg["disambiguator_timeout"]),
        )
    except Exception as e:
        logger.warning("entity_disambiguator_failed", extra={"error": describe_error(e)})
        lf_mark_current("WARNING", f"disambiguator call failed ({describe_error(e)}); fallback")
        await log(status="failed", error_code=type(e).__name__, error_message=describe_error(e))
        return None

    observe_llm_latency(model_id, "entity_disambiguator", resp.stats)
    await log(status="completed", **stats_to_request_kwargs(resp.stats))
    try:
        picks = _Picks.model_validate_json(resp.text or "")
    except ValidationError:
        logger.warning("entity_disambiguator_unparseable", extra={"raw": (resp.text or "")[:300]})
        lf_mark_current("WARNING", "disambiguator output unparseable; fallback")
        return None
    return _map(picks, entities, candidates)
