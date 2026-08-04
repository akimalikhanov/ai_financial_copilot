"""FindingsLedger — the run's keyed store of concluded findings (the scratchpad).

The third state store, beside `Transcript` (the model's view) and `EvidenceLedger`
(retrieved chunks). Where the transcript logs *interactions*, this stores *conclusions*,
addressed by a stable key — ``EntityFinding.entity`` or ``Observation.aspect`` — and
updated in place.

10a (this step): the loop populates it by parsing every finalizer attempt (accepted or
rejected) via `ingest`, and projects it back for synthesis via `projection`. No
model-facing ``record_*`` tools and no prompt change — that is 10b (Patterns 1/3).
Because the ledger's readers (synthesis now; gates/carry-over later) are identical
whether the loop or the model writes it, this seam makes those a behaviour addition, not
a rewrite.

Contract C4: best-per-aspect by construction — `record` updates in place, so there is no
best-of comparator. Contract C6: an item whose chunk refs don't resolve in the
`EvidenceLedger` is dropped, not admitted (the model cannot land an ungrounded
conclusion); the prior entry for that key is left intact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal
from uuid import UUID

from src.schemas.agent_findings import (
    AgentFindings,
    AnalyticalFindings,
    EntityFinding,
    Observation,
)

if TYPE_CHECKING:
    from src.services.chat.agent.evidence import EvidenceLedger

_DEGRADED_CAVEAT = (
    "The search did not fully converge; these findings are partial and may be incomplete."
)


@dataclass
class FindingEntry:
    key: str
    finding: EntityFinding | Observation
    revisions: int = 0  # 0 on first write, ++ on each in-place update


def _resolves(refs: list[str], evidence: EvidenceLedger) -> bool:
    """True if at least one ref names a chunk present in the evidence ledger.

    Refs are chunk-UUID strings by `ingest` time (the loop resolves S-labels to UUIDs
    and drops unresolvable ones before recording).
    """
    for ref in refs:
        try:
            if UUID(ref) in evidence:
                return True
        except ValueError:
            continue
    return False


def _is_grounded(finding: EntityFinding | Observation, evidence: EvidenceLedger) -> bool:
    if isinstance(finding, EntityFinding):
        # A negative finding ("not available") legitimately cites nothing; only a
        # positive value claim must be grounded.
        if not finding.available:
            return True
        return _resolves(finding.source_chunks, evidence)
    return _resolves([*finding.evidence_chunks, *(finding.refuted_by or [])], evidence)


class FindingsLedger:
    def __init__(self) -> None:
        self._entries: dict[str, FindingEntry] = {}
        self._kind: Literal["agent", "analytical"] | None = None
        # Finalizer-envelope metadata, last-write-wins — the per-item entries alone
        # cannot reconstruct the AgentFindings/AnalyticalFindings shape synthesis and
        # persistence expect, so the envelope is retained here rather than re-derived.
        self._metric_requested: str | None = None
        self._comparison_op: Literal["argmin", "argmax", "list", "none"] | None = None
        self._question: str | None = None
        self._conclusion: str | None = None
        self._gaps: list[str] | None = None

    def record(
        self, key: str, finding: EntityFinding | Observation, evidence: EvidenceLedger
    ) -> bool:
        """Insert or update-in-place. Returns False (and leaves any prior entry intact)
        when C6's grounding filter drops the item, or when the item's type disagrees with
        the kind already established for this run."""
        kind: Literal["agent", "analytical"] = (
            "agent" if isinstance(finding, EntityFinding) else "analytical"
        )
        # Both finalizers are offered on every request, so a stray off-kind call would
        # otherwise flip _kind and make projection() drop every entry of the real kind.
        if self._kind is not None and kind != self._kind:
            return False
        if not _is_grounded(finding, evidence):
            return False
        self._kind = kind
        entry = self._entries.get(key)
        if entry is None:
            self._entries[key] = FindingEntry(key=key, finding=finding)
        else:
            entry.finding = finding
            entry.revisions += 1
        return True

    def ingest(
        self, candidate: AgentFindings | AnalyticalFindings, evidence: EvidenceLedger
    ) -> None:
        """Fold one finalizer attempt (accepted or rejected) into the ledger, without pruning.

        Restated keys update in place (revisions++); keys this attempt omits are left
        alone. Pruning happens only in `prune_to`, called on the attempt that is actually
        *accepted* — a rejected attempt is not a deliberate restatement, it's a draft the
        model is about to be told to redo, and treating its omissions as abandonment
        destroys established entries before any gate has even evaluated them.

        Envelope fields are last-write-wins but null-guarded: a later attempt that omits a
        field must not erase the value an earlier one established.
        """
        kind: Literal["agent", "analytical"] = (
            "agent" if isinstance(candidate, AgentFindings) else "analytical"
        )
        if self._kind is not None and kind != self._kind:
            return
        items: list[tuple[str, EntityFinding | Observation]]
        if isinstance(candidate, AgentFindings):
            self._kind = "agent"
            self._metric_requested = candidate.metric_requested or self._metric_requested
            self._comparison_op = candidate.comparison_op or self._comparison_op
            items = [(f.entity, f) for f in candidate.findings]
        else:
            self._kind = "analytical"
            self._question = candidate.question or self._question
            self._conclusion = candidate.conclusion
            self._gaps = list(candidate.gaps) if candidate.gaps else None
            items = [(o.aspect, o) for o in candidate.observations]
        for key, finding in items:
            self.record(key, finding, evidence)

    def prune_to(self, candidate: AgentFindings | AnalyticalFindings) -> set[str]:
        """Drop keys the *accepted* restatement abandoned. Returns the dropped keys.

        A finalizer is a complete re-statement, so on the accepted call a key the model
        omitted was deliberately abandoned and must not be resurrected into the served
        projection (the same reason `transcript.py` strips rejected drafts). The caller
        records the dropped keys as a gap, so the omission at least reaches the answer as
        a stated limitation rather than vanishing.
        """
        if isinstance(candidate, AgentFindings):
            live = {f.entity for f in candidate.findings}
        else:
            live = {o.aspect for o in candidate.observations}
        dropped = self._entries.keys() - live
        for key in dropped:
            del self._entries[key]
        return dropped

    def add_gap(self, gap: str) -> None:
        """Append a loop-authored caveat to the served envelope's `gaps`."""
        self._gaps = [*(self._gaps or []), gap]

    def keys(self) -> set[str]:
        return set(self._entries)

    def entry(self, key: str) -> FindingEntry | None:
        return self._entries.get(key)

    def projection(self, *, degraded: bool = False) -> AgentFindings | AnalyticalFindings | None:
        """The findings synthesis serves. None when no finalizer was ever attempted
        (raw-excerpt fallback); otherwise the accumulated ledger reconstructed into its
        finalizer shape, marked degraded when the run never sealed (subsumes the old
        best-effort path, P1-6). Degraded only annotates the analytical path, whose
        `gaps` field can carry the caveat."""
        if self._kind == "agent":
            findings = tuple(
                e.finding for e in self._entries.values() if isinstance(e.finding, EntityFinding)
            )
            return AgentFindings(
                metric_requested=self._metric_requested or "",
                findings=findings,
                comparison_op=self._comparison_op,
            )
        if self._kind == "analytical":
            observations = tuple(
                e.finding for e in self._entries.values() if isinstance(e.finding, Observation)
            )
            gaps = list(self._gaps) if self._gaps else []
            if degraded:
                gaps.append(_DEGRADED_CAVEAT)
            return AnalyticalFindings(
                question=self._question or "",
                observations=observations,
                conclusion=self._conclusion,
                gaps=gaps or None,
            )
        return None
