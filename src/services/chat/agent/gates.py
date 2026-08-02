"""Finalizer gates: candidate parsing/prep helpers, the two current gates, and the
single shared rejection routine both go through.

Gate = Callable[[Candidate, AgentRunState], Rejection | None] — a `Rejection` describing
*what to charge and how to label it*, or None to allow. Registered per tool in `tools.py`
(Contract C3: structural before sufficiency); `loop.py` runs them in order and stops at
the first rejection.

Gates are pure predicates (P1-4): they never mutate `state`. Every counter a rejection
spends is declared on the `Rejection` and applied once, in `reject()` — so a speculative
or retried gate evaluation cannot silently burn budget, and `loop.py` needs no knowledge
of which gate fired.

This directly fixes P1-8: both rejection paths now go through the same `reject()`,
so both parse their candidate (feeding the `FindingsLedger` via `ingest` and `protect`)
and both compress history — previously only the analytical path did either.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from redis.asyncio import Redis

from src.observability import langfuse as lf_client
from src.observability.metrics import AGENT_TOOL_CALLS
from src.redis_client import add_event
from src.schemas.agent_findings import (
    AgentFindings,
    AnalyticalFindings,
    Observation,
    describe_key,
    normalize_item_name,
    observation_key,
)
from src.services.chat.agent.transcript import stub_rejected_tool_call
from src.services.llm_adapters.base_adapter import ChatMessage, Role, ToolCallRef

if TYPE_CHECKING:
    from src.services.chat.agent.state import AgentRunState

logger = logging.getLogger(__name__)

Candidate = AgentFindings | AnalyticalFindings


@dataclass(frozen=True)
class Rejection:
    """A gate's verdict: why to reject, how to label it, and which budgets it spends.

    Everything here is *declarative* — `reject()` is the only writer. `metric_status`
    distinguishes the rejection on the `AGENT_TOOL_CALLS` status label, the Langfuse span
    name, and the SSE reason (D15). `charges_insufficiency` and `named_item_keys` name
    the budgets: the two are deliberately disjoint pools (D1/FR-5), since
    `max_insufficiency_rejections` defaults to 1 and would otherwise be drained by a
    single find-then-follow round.
    """

    reason: str
    metric_status: str = "rejected"
    charges_insufficiency: bool = False
    charges_restatement: bool = False
    # Normalized keys charged against the per-item pool, and the model's own wording for
    # the same items — the keys bill, the names are quoted back at the model (D15).
    named_item_keys: tuple[str, ...] = field(default_factory=tuple)
    named_item_names: tuple[str, ...] = field(default_factory=tuple)


GateFn = Callable[[Candidate, "AgentRunState"], Rejection | None]


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
        key = normalize_item_name(item.name)
        if not key:
            continue
        if item.status == "unresolved":
            open_items.setdefault(key, (item.name, o.aspect))
        else:
            closed.add(key)
    return {k: v for k, v in open_items.items() if k not in closed}


def missing_entity_gate(candidate: Candidate, state: AgentRunState) -> Rejection | None:
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
    return Rejection(reason="report_findings is incomplete — " + "; ".join(parts) + ".")


def analytical_insufficiency_gate(candidate: Candidate, state: AgentRunState) -> Rejection | None:
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
    reason = _analytical_insufficiency(candidate)
    return None if reason is None else Rejection(reason=reason, charges_insufficiency=True)


def named_item_gate(candidate: Candidate, state: AgentRunState) -> Rejection | None:
    """Sequential-depth gate: reject an analytical finalizer that names an item without
    reporting its value, while budget remains to go find it.

    Unresolved items are read from **this attempt's observations only** (P2-1). A
    finalizer is a complete restatement, so the candidate is the authoritative statement
    of what is still open; the ledger's projection would additionally carry keys this
    attempt abandoned, and re-flagging those here would fight `restatement_integrity_gate`,
    which is the mechanism that actually owns abandonment.

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

    unresolved = unresolved_named_items(candidate.observations)

    # FR-6 per-item: an item that has already been chased to its cap stops causing
    # rejection on its own account, and stops consuming the shared pool with it (§3).
    chargeable = {
        key: value
        for key, value in unresolved.items()
        if state.named_item_rejections.get(key, 0) < state.effort.max_named_item_rejections_per_item
    }
    if not chargeable:
        return None

    items = "; ".join(f'"{name}" (aspect: {aspect})' for name, aspect in chargeable.values())
    return Rejection(
        reason=(
            f"these named items were reported without their value: {items}. Call "
            "search_documents for each, combining the item name with its aspect. If that "
            "search shows the documents do not disclose the value, re-report that "
            'observation with named_item.status = "confirmed_absent".'
        ),
        metric_status="rejected_named_item",
        named_item_keys=tuple(chargeable),
        named_item_names=tuple(name for name, _ in chargeable.values()),
    )


def confirmed_absent_gate(candidate: Candidate, state: AgentRunState) -> Rejection | None:
    """Reject a `confirmed_absent` status the run has no search to back (P1-2).

    The v4 prompt states the precondition — an item may only be closed absent once a
    `search_documents` query has contained its name — but a stated precondition is not a
    checked one, and closing an item early is the cheaper move under budget pressure. A
    false `confirmed_absent` is a fabricated non-disclosure finding, so this checks it
    against `state.issued_queries`, the verbatim queries the loop actually ran.

    Substring match on the normalized item key, consistent with the deliberate decision
    not to fuzzy-match item names (FR-12): a miss costs one extra turn, a false pass costs
    a wrong answer. Charged to the same per-item pool as `named_item_gate`, so one item
    cannot exceed its cap by alternating between the two failure modes.
    """
    if not isinstance(candidate, AnalyticalFindings):
        return None
    if state.named_item_rejections_total >= state.effort.max_named_item_rejections_total:
        return None
    if state.iteration >= state.effort.max_iterations - 1 or not state.spend_within_budget():
        return None

    unverified: dict[str, str] = {}
    for o in candidate.observations:
        item = o.named_item
        if item is None or item.status != "confirmed_absent":
            continue
        key = normalize_item_name(item.name)
        if not key or key in unverified:
            continue
        if any(key in q for q in state.issued_queries):
            continue
        if (
            state.named_item_rejections.get(key, 0)
            >= state.effort.max_named_item_rejections_per_item
        ):
            continue
        unverified[key] = item.name
    if not unverified:
        return None

    items = ", ".join(f'"{name}"' for name in unverified.values())
    return Rejection(
        reason=(
            f'these items were marked "confirmed_absent" but no search this run named '
            f"them: {items}. Absence is only established by a search that names the item. "
            "Call search_documents for each, using the item's exact name in the query, "
            "then re-report — resolved if the figure turns up, confirmed_absent if it "
            "does not. Restate every other observation unchanged."
        ),
        metric_status="rejected_named_item",
        named_item_keys=tuple(unverified),
        named_item_names=tuple(unverified.values()),
    )


# Placeholder figures are never legitimate in a claim and are lexically detectable:
# a currency symbol followed by X's, an all-X percentage, or a bare "N <scale>". Kept
# case-sensitive — uppercase X/N is the placeholder form, and lowercasing would start
# matching ordinary prose.
_PLACEHOLDER_RE = re.compile(
    r"[¥$€£₩]\s*X+|\bX+(?:[.,]X+)?\s*%|\bN\s+(?:thousand|million|billion|trillion)\b"
)


def restatement_integrity_gate(candidate: Candidate, state: AgentRunState) -> Rejection | None:
    """Reject a restatement that degrades or discards content the ledger already holds (P0-1).

    Two checks, one budget:

    *Placeholders* — a claim containing `¥XXX`, `X%` or `N billion` is a figure the model
    re-derived from memory and could not recall. Checked on every attempt, since there is
    no legitimate use, and it catches degradation the key-set check below cannot see
    (a same-key restatement that vaguens a precise figure).

    *Key loss* — after a rejection has forced a full restatement, any aspect key that was
    in the ledger and is not in this candidate has been dropped or renamed. `ingest` folds
    rejected attempts in without pruning (see `findings.py`), so `keys_before_attempt` —
    snapshotted by the loop before the fold — is the referent: by gate time the ledger
    itself no longer distinguishes established keys from this attempt's.

    Restricted to attempts that follow a prior rejection: a first, voluntary finalizer has
    nothing established to lose, and deliberate abandonment on an accepted call stays
    legal once the budget is spent (the loop records it as a gap instead, P1-3).
    """
    if not isinstance(candidate, AnalyticalFindings):
        return None
    if state.restatement_rejections >= state.effort.max_restatement_rejections:
        return None
    if state.iteration >= state.effort.max_iterations - 1 or not state.spend_within_budget():
        return None

    for o in candidate.observations:
        if _PLACEHOLDER_RE.search(o.claim):
            return Rejection(
                reason=(
                    f"observation '{o.aspect}' states a placeholder instead of a figure: "
                    f"'{o.claim[:120]}'. Re-report it with the real number from the "
                    "excerpts, or state plainly that the documents do not disclose it. "
                    "Never emit a stand-in figure."
                ),
                metric_status="rejected_restatement",
                charges_restatement=True,
            )

    if not state.finalizer_rejections:
        return None
    dropped = state.keys_before_attempt - {observation_key(o) for o in candidate.observations}
    if not dropped:
        return None
    return Rejection(
        reason=(
            "this restatement dropped observations you had already established: "
            f"{', '.join(describe_key(k) for k in sorted(dropped))}. A rejection concerns "
            "only the items it names — every other observation must come back under its "
            "ORIGINAL aspect key, naming the same item, with its figures intact. Re-emit "
            "them from the established list below, then apply your updates."
        ),
        metric_status="rejected_restatement",
        charges_restatement=True,
    )


def _established_block(state: AgentRunState) -> str:
    """The ledger's grounded projection, rendered in the shape the model must re-emit (P0-2).

    The rejection stub deletes the rejected call's arguments from history and compaction
    evicts the excerpts, so "copy your other observations across verbatim" asks the model
    to copy from a blank page — which is how established figures get re-derived and
    degraded. This hands the content back.

    Safe rather than a reversal of the stub decision, on two counts. It is the *projection*
    — per-key collapsed and grounding-filtered, so an ungrounded draft claim was already
    dropped and is not offered back for copying. And it lands on the tool result, which
    compaction can evict and which reads as system-supplied state, rather than on the
    assistant turn, which survives compaction and would train the model to imitate its own
    output shape.
    """
    prior = state.findings.projection()
    if not isinstance(prior, AnalyticalFindings) or not prior.observations:
        return ""
    lines: list[str] = []
    for o in prior.observations:
        # The ledger stores resolved chunk UUIDs; the model must emit the short excerpt
        # labels it was shown, so map back through the evidence ledger.
        labels = ", ".join(f'"{label}"' for label in state.evidence.labels_for(o.evidence_chunks))
        item = ""
        if o.named_item is not None:
            item = f', named_item: {{name: "{o.named_item.name}", status: "{o.named_item.status}"}}'
        lines.append(
            f'- {{aspect: "{o.aspect}", claim: "{o.claim}", '
            f'evidence_chunks: [{labels}], confidence: "{o.confidence}"{item}}}'
        )
    return (
        "\n\nEstablished so far — your next call must contain these observations, "
        "unchanged, plus your updates:\n" + "\n".join(lines)
    )


def _apply_charges(state: AgentRunState, rejection: Rejection) -> None:
    """Spend the budgets a `Rejection` declares. Called only from `reject()` — gates
    themselves must stay pure (P1-4)."""
    state.finalizer_rejections += 1
    if rejection.charges_insufficiency:
        state.insufficiency_rejections += 1
    if rejection.charges_restatement:
        state.restatement_rejections += 1
    for key in rejection.named_item_keys:
        state.named_item_rejections[key] = state.named_item_rejections.get(key, 0) + 1
    state.named_item_rejections_total += len(rejection.named_item_keys)
    if rejection.named_item_keys:
        # P0-4: arm one empty-round exemption. A named-item follow-up is a *narrowing*
        # search over ground an earlier hop already covered, so returning no new chunks is
        # its normal outcome — and is precisely the evidence that licenses
        # `confirmed_absent`. Without this, the empty-round nudge ("do not finalize, search
        # elsewhere") contradicts the rejection ("re-report it as confirmed absent") in the
        # one scenario the mechanism exists for.
        state.named_item_grace = rejection.named_item_names


async def reject(
    *,
    rejection: Rejection,
    finalizer_tc: ToolCallRef,
    candidate: Candidate,
    state: AgentRunState,
    redis_app: Redis,
    request_id: str,
) -> None:
    """Shared rejection routine — the single writer of every counter a rejection spends.

    Stubs the tool call, appends the rejection message (with the established-findings
    block, P0-2), charges the budgets the `Rejection` declares, increments the metric,
    emits the LF span and SSE event, and compresses history.

    Charging here rather than inside the gates (P1-4) means a gate can be evaluated
    speculatively without cost, and `loop.py` treats every gate identically instead of
    branching on gate identity to decide the metric label and the budget.
    """
    reason = rejection.reason
    state.transcript.append_tool_calls([stub_rejected_tool_call(finalizer_tc)])
    state.transcript.append(
        ChatMessage(
            role=Role.tool,
            tool_call_id=finalizer_tc.id,
            content=f"{finalizer_tc.name} rejected — {reason}{_established_block(state)}",
        )
    )
    AGENT_TOOL_CALLS.labels(finalizer_tc.name, rejection.metric_status).inc()

    lf = lf_client.get_client()
    if lf:
        span_name = f"{finalizer_tc.name}_{rejection.metric_status}"
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
            "reason": f"{finalizer_tc.name}_{rejection.metric_status}",
        },
    )

    _apply_charges(state, rejection)
    state.transcript.compress()
