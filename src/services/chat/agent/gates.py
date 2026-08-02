"""Finalizer gates: candidate parsing/prep helpers, the two current gates, and the
single shared rejection routine both go through.

Gate = Callable[[Candidate, AgentRunState], str | None] — a reason to reject, or None
to allow. Registered per tool in `tools.py` (Contract C3: structural before
sufficiency); `loop.py` runs them in order and stops at the first reason.

This directly fixes P1-8: both rejection paths now go through the same `reject()`,
so both parse their candidate (feeding the `FindingsLedger` via `ingest` and `protect`)
and both compress history — previously only the analytical path did either.
"""

from __future__ import annotations

import logging
import string
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

from redis.asyncio import Redis

from src.observability import langfuse as lf_client
from src.observability.metrics import AGENT_TOOL_CALLS
from src.redis_client import add_event
from src.schemas.agent_findings import AgentFindings, AnalyticalFindings, Observation
from src.services.chat.agent.transcript import stub_rejected_tool_call
from src.services.llm_adapters.base_adapter import ChatMessage, Role, ToolCallRef

if TYPE_CHECKING:
    from src.services.chat.agent.state import AgentRunState

logger = logging.getLogger(__name__)

Candidate = AgentFindings | AnalyticalFindings
GateFn = Callable[[Candidate, "AgentRunState"], str | None]


def parse_findings(tc: ToolCallRef) -> Candidate:
    """Parse and validate a finalizer tool call against its Pydantic schema.

    Raises ``pydantic.ValidationError`` on malformed JSON or a schema violation,
    surfaced at the caller instead of silently constructing garbage via ``.get()``.
    """
    if tc.name == "report_findings":
        return AgentFindings.model_validate_json(tc.arguments)
    return AnalyticalFindings.model_validate_json(tc.arguments)


def finding_chunk_ids(findings: Candidate) -> set[str]:
    """All chunk-id strings referenced by the findings."""
    ids: set[str] = set()
    if isinstance(findings, AgentFindings):
        for f in findings.findings:
            ids.update(f.source_chunks or [])
    else:
        for o in findings.observations:
            ids.update(o.evidence_chunks or [])
            ids.update(o.refuted_by or [])
    return ids


def drop_evidence_free_observations(findings: AnalyticalFindings) -> AnalyticalFindings:
    """Route observations with no resolvable evidence_chunks into gaps instead of synthesis.

    The analytical insufficiency gate rejects these while iteration/rejection budget
    remains, but the gate is skipped once that budget runs out, and ref resolution can
    also empty out a previously non-empty evidence_chunks list. Either way, an uncited
    claim must not reach synthesis looking like a settled fact.
    """
    kept: list[Observation] = []
    dropped_claims: list[str] = []
    for o in findings.observations:
        if o.evidence_chunks:
            kept.append(o)
        else:
            dropped_claims.append(o.claim)
    if not dropped_claims:
        return findings
    gaps = list(findings.gaps or []) + [f"Unsubstantiated claim: {c}" for c in dropped_claims]
    return findings.model_copy(update={"observations": tuple(kept), "gaps": gaps})


def _analytical_insufficiency(findings: AnalyticalFindings) -> str | None:
    """Return a rejection reason if analytical findings are too thin to finalize, else None.

    Reads the sufficiency signals the model already writes (Pattern 3, 3.ii): a finalizer
    attempt is insufficient when it rests on low-confidence evidence or declares open gaps.
    This is a deterministic gate — the irreducible-judgment LLM evaluator (3.iii) is only
    warranted where this rule demonstrably under-fires.
    """
    if not findings.observations:
        return "No observations were reported. Search for supporting evidence before finalizing."
    for o in findings.observations:
        if not o.evidence_chunks:
            return (
                f"Observation '{o.claim[:80]}' cites no evidence_chunks — every claim must "
                "reference at least one supporting chunk before finalizing."
            )
    if findings.gaps:
        return (
            "Open gaps remain: "
            + "; ".join(findings.gaps)
            + ". Search to close these gaps before finalizing."
        )
    if all(o.confidence == "low" for o in findings.observations):
        return (
            "Every observation is low-confidence. Search differently — likely a footnote, "
            "reconciliation, or segment table — to corroborate before finalizing."
        )
    return None


_LEADING_ARTICLES = ("the ", "a ", "an ")


def _normalize_item_name(name: str) -> str:
    """D9's identity key: case-fold, strip surrounding whitespace/punctuation, drop a
    leading article, collapse internal whitespace.

    Deterministic string handling only — no similarity computation (FR-12). Two names
    differing by more than case, surrounding punctuation, leading article, or internal
    spacing are simply different items; canonical-name reuse is a prompt-enforced
    discipline (D9, EC-5), not something this function tries to recover.
    """
    key = " ".join(name.split()).strip().casefold()
    key = key.strip(string.punctuation + string.whitespace)
    for article in _LEADING_ARTICLES:
        if key.startswith(article):
            key = key[len(article) :]
            break
    return " ".join(key.split())


def unresolved_named_items(
    observations: Iterable[Observation],
) -> dict[str, tuple[str, str]]:
    """Named items still open across ``observations``: normalized key -> (name, aspect).

    A key is unresolved when at least one observation reports it ``unresolved`` and no
    observation reports it ``resolved`` or ``confirmed_absent`` (FR-2, FR-2a, EC-2).
    Resolution wins regardless of observation order (FR-8), so this is computed in two
    passes rather than by last-write-wins.

    ``confirmed_absent`` closes an item exactly as ``resolved`` does — the documents not
    disclosing a value is an answer, not a gap to keep chasing (FR-2a). No value
    comparison anywhere: two ``resolved`` observations carrying different values are both
    simply resolved, and adjudicating them is not this function's job (EC-11, D12).

    ``name``/``aspect`` come from the first observation to report the key unresolved, so
    a rejection message quotes the model's own wording back to it (D15).
    """
    open_items: dict[str, tuple[str, str]] = {}
    closed: set[str] = set()
    for o in observations:
        item = o.named_item
        if item is None:
            continue
        key = _normalize_item_name(item.name)
        if not key:
            continue
        if item.status == "unresolved":
            open_items.setdefault(key, (item.name, o.aspect))
        else:
            closed.add(key)
    return {k: v for k, v in open_items.items() if k not in closed}


def missing_entity_gate(candidate: Candidate, state: AgentRunState) -> str | None:
    """Structural gate: reject report_findings until every expected entity has been both
    *searched* and *reported* as a finding.

    Reading ``findings.keys()`` (populated by the loop's `ingest` before gates run) —
    not just ``searched_entities`` — closes the gap where an entity was searched but
    omitted from the report: it previously slipped past both this gate and synthesis'
    unsearched-stub backstop, vanishing from the answer. Keeping the searched check too
    preserves the "must actually retrieve" guarantee, so an entity can't be waved
    through by reporting ``available=false`` without ever searching. An entity for which
    the documents genuinely lack the value is covered by reporting it ``available=false``
    (grounding-exempt, so it still takes a ledger key).

    Unconditional — unlike the sufficiency gate below, this never waives on iteration or
    budget pressure: a request missing coverage should not finalize regardless of budget.
    """
    if not isinstance(candidate, AgentFindings) or not state.expected_entities:
        return None
    unsearched = state.expected_entities - state.searched_entities
    unreported = (state.expected_entities & state.searched_entities) - state.findings.keys()
    if not unsearched and not unreported:
        return None
    parts: list[str] = []
    if unsearched:
        parts.append(f"search these entities first: {', '.join(sorted(unsearched))}")
    if unreported:
        parts.append(
            "report a finding for each already-searched entity (use available=false when "
            f"the value isn't in the documents): {', '.join(sorted(unreported))}"
        )
    return "report_findings is incomplete — " + "; ".join(parts) + "."


def analytical_insufficiency_gate(candidate: Candidate, state: AgentRunState) -> str | None:
    """Sufficiency gate: reject a thin analytical finalizer while retry budget remains.

    3.ii/3.iii — re-prompt with the specific gap instead of finalizing on low-confidence
    evidence. Only fires when a further round is possible: iteration budget remains,
    token budget remains, and the gate hasn't already rejected past its per-request cap
    (uncapped rejection loops drove iteration_cap / high spend with little gain).
    """
    if not isinstance(candidate, AnalyticalFindings):
        return None
    can_retry = (
        state.iteration < state.effort.max_iterations - 1
        and state.spend_within_budget()
        and state.insufficiency_rejections < state.effort.max_insufficiency_rejections
    )
    if not can_retry:
        return None
    return _analytical_insufficiency(candidate)


def named_item_gate(candidate: Candidate, state: AgentRunState) -> str | None:
    """Sequential-depth gate: reject an analytical finalizer that names an item without
    reporting its value, while budget remains to go find it.

    Unresolved items are computed over the **union** of this attempt's observations and
    the ledger's projection. The candidate is pre-collapse and complete; the projection
    carries prior attempts' surviving state. Reading the projection alone would let a
    resolved-then-unresolved ordering under one aspect discard the resolution and re-flag
    a settled item, since `ingest()` collapses same-aspect observations last-write-wins.

    Waives on the same retry predicate as `analytical_insufficiency_gate` — there is no
    point rejecting when no further round can happen (FR-6, EC-4). Deliberately does not
    read `insufficiency_rejections`: the two gates hold independent budgets (D1).
    """
    if not isinstance(candidate, AnalyticalFindings):
        return None
    if state.named_item_rejections_total >= state.effort.max_named_item_rejections_total:
        return None
    can_retry = state.iteration < state.effort.max_iterations - 1 and state.spend_within_budget()
    if not can_retry:
        return None

    prior = state.findings.projection()
    prior_observations = prior.observations if isinstance(prior, AnalyticalFindings) else ()
    unresolved = unresolved_named_items((*candidate.observations, *prior_observations))

    # FR-6 per-item: an item that has already been chased to its cap stops causing
    # rejection on its own account, and stops consuming the shared pool with it (§3).
    chargeable = {
        key: value
        for key, value in unresolved.items()
        if state.named_item_rejections.get(key, 0) < state.effort.max_named_item_rejections_per_item
    }
    if not chargeable:
        return None

    for key in chargeable:
        state.named_item_rejections[key] = state.named_item_rejections.get(key, 0) + 1
    state.named_item_rejections_total += len(chargeable)

    items = "; ".join(f'"{name}" (aspect: {aspect})' for name, aspect in chargeable.values())
    return (
        f"these named items were reported without their value: {items}. Call "
        "search_documents for each, combining the item name with its aspect. If a search "
        "shows the documents do not disclose the value, re-report that observation with "
        'named_item.status = "confirmed_absent". Restate every already-resolved named item '
        "in your next report_analytical_findings call."
    )


async def reject(
    *,
    reason: str,
    finalizer_tc: ToolCallRef,
    candidate: Candidate,
    state: AgentRunState,
    redis_app: Redis,
    request_id: str,
    metric_status: str = "rejected",
    charge_insufficiency: bool = True,
) -> None:
    """Shared rejection boilerplate: stub the tool call, append the rejection message,
    increment the metric, emit the LF span and SSE event, compress history.

    `metric_status` distinguishes *why* a finalizer was rejected on the existing
    `AGENT_TOOL_CALLS` status label, the LF span name, and the SSE reason — no new
    metric object (D15). Defaults to the previous constant, so existing callers are
    behaviour-neutral.

    `charge_insufficiency` names which budget this rejection spends. It must be False
    for a `named_item_gate` rejection: D1/FR-5 require the two gates' budgets be
    independent, and `max_insufficiency_rejections` defaults to 1, so charging a
    named-item rejection to it would disable `analytical_insufficiency_gate` for the
    rest of the run after a single find-then-follow round.
    """
    state.transcript.append_tool_calls([stub_rejected_tool_call(finalizer_tc)])
    state.transcript.append(
        ChatMessage(
            role=Role.tool,
            tool_call_id=finalizer_tc.id,
            content=f"{finalizer_tc.name} rejected — {reason}",
        )
    )
    AGENT_TOOL_CALLS.labels(finalizer_tc.name, metric_status).inc()

    lf = lf_client.get_client()
    if lf:
        span_name = f"{finalizer_tc.name}_{metric_status}"
        lf_input: dict = {"reason": reason}
        if isinstance(candidate, AnalyticalFindings):
            lf_input = {
                "confidence": [o.confidence for o in candidate.observations],
                "gaps": candidate.gaps,
            }
        with lf.start_as_current_observation(
            as_type="span", name=span_name, input=lf_input
        ) as span:
            span.update(output={"reason": reason}, metadata={"iteration": state.iteration})

    await add_event(
        redis_app,
        request_id,
        "tool_call_completed",
        {
            "entity": "__finalizer__",
            "error": True,
            "reason": f"{finalizer_tc.name}_{metric_status}",
        },
    )
    if charge_insufficiency and isinstance(candidate, AnalyticalFindings):
        state.insufficiency_rejections += 1
    state.transcript.compress()
