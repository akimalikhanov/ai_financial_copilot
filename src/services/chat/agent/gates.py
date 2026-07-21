"""Finalizer gates: candidate parsing/prep helpers, the two current gates, and the
single shared rejection routine both go through.

Gate = Callable[[Candidate, AgentRunState], str | None] — a reason to reject, or None
to allow. Registered per tool in `tools.py` (Contract C3: structural before
sufficiency); `loop.py` runs them in order and stops at the first reason.

This directly fixes P1-8: both rejection paths now go through the same `reject()`,
so both parse their candidate (feeding `last_candidate`/`protect`) and both compress
history — previously only the analytical path did either.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
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


def missing_entity_gate(candidate: Candidate, state: AgentRunState) -> str | None:
    """Structural gate: reject report_findings if any expected entity was never searched.

    Unconditional — unlike the sufficiency gate below, this never waives on iteration
    or budget pressure: a request that never searched an expected entity should not
    finalize on it regardless of how much budget remains.
    """
    if not isinstance(candidate, AgentFindings) or not state.expected_entities:
        return None
    missing = state.expected_entities - state.searched_entities
    if not missing:
        return None
    return (
        f"missing searches for: {', '.join(sorted(missing))}. "
        "Search each missing entity before calling report_findings."
    )


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


async def reject(
    *,
    reason: str,
    finalizer_tc: ToolCallRef,
    candidate: Candidate,
    state: AgentRunState,
    redis_app: Redis,
    request_id: str,
) -> None:
    """Shared rejection boilerplate: stub the tool call, append the rejection message,
    increment the metric, emit the LF span and SSE event, compress history."""
    state.transcript.append_tool_calls([stub_rejected_tool_call(finalizer_tc)])
    state.transcript.append(
        ChatMessage(
            role=Role.tool,
            tool_call_id=finalizer_tc.id,
            content=f"{finalizer_tc.name} rejected — {reason}",
        )
    )
    AGENT_TOOL_CALLS.labels(finalizer_tc.name, "rejected").inc()

    lf = lf_client.get_client()
    if lf:
        span_name = f"{finalizer_tc.name}_rejected"
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
        {"entity": "__finalizer__", "error": True, "reason": f"{finalizer_tc.name}_rejected"},
    )
    state.insufficiency_rejections += 1 if isinstance(candidate, AnalyticalFindings) else 0
    state.transcript.compress()
