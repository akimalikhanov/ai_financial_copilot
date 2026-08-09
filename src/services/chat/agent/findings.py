"""FindingsLedger — the run's keyed store of concluded findings (the scratchpad).

The third state store, beside `Transcript` (the model's view) and `EvidenceLedger`
(retrieved chunks). Where the transcript logs *interactions*, this stores *conclusions*,
addressed by a stable key — ``EntityFinding.entity`` or ``Observation.aspect`` — and
updated in place.

Post-D3 the loop folds in every `report_*` call as it arrives — reports are incremental,
not terminal — and projects the accumulation back for synthesis via `projection`.

Contract C4: best-per-aspect by construction — `record` updates in place, so there is no
best-of comparator. Contract C6: an item whose chunk refs don't resolve in the
`EvidenceLedger` is dropped, not admitted (the model cannot land an ungrounded
conclusion); the prior entry for that key is left intact.

With the gates deleted, C6's grounding filter and `drop_evidence_free_observations` are
the *only* remaining correctness filters on what reaches synthesis.
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

Candidate = AgentFindings | AnalyticalFindings

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
    # The analytical counterpart: a stated negative ("searched, the documents don't
    # disclose this") is a real conclusion about its aspect and cites nothing by
    # definition. Without this it could only be expressed as an unkeyed gap string, which
    # closes no aspect — so the loop kept searching what the model had already settled.
    if not finding.substantiated:
        return True
    return _resolves([*finding.evidence_chunks, *(finding.refuted_by or [])], evidence)


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
    """Drop observations that assert a substantiated claim while citing nothing.

    Post-D3 this and `_is_grounded` are the only correctness filters on what reaches
    synthesis — the gates are gone, so grounding carries the whole load. An observation
    citing `refuted_by` but no `evidence_chunks` is legitimately grounded (a refutation is
    a finding), matching `_is_grounded`'s either-list rule.

    `substantiated=False` is exempt: that is a stated negative, which cites nothing by
    design and is kept as a real entry under its aspect key. So what remains here is only
    the pure hallucination case — a claim asserting support it never produced. Dropping it
    writes no gap: D4 reconciliation in `_apply_report` already closes the key, and the
    claim text is exactly what must not reach a user-facing caveat.
    """
    kept = tuple(
        o for o in findings.observations if o.evidence_chunks or o.refuted_by or not o.substantiated
    )
    if len(kept) == len(findings.observations):
        return findings
    return findings.model_copy(update={"observations": kept})


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
        # D4: keys whose only output is a stated gap, keyed to the reason text that
        # closed them (not just membership) — so a caller (e.g. retry logic) can join
        # this against `AgentRunState.aspect_stats` without pattern-matching `_gaps`
        # strings. `addressed` is derived from keys() | closed_as_gap(), so an aspect
        # can never close silently.
        self._closed_as_gap: dict[str, str] = {}

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
        """Fold one report into the ledger. Accumulates; never prunes.

        Restated keys update in place (revisions++); keys this report omits are left
        alone. Post-D3 reports are incremental rather than a single terminal restatement,
        so omission carries no information at all — the model reports an aspect when its
        evidence settles and never restates the others.

        Envelope fields are last-write-wins but null-guarded, and `gaps` unions rather than
        replaces: a later report that omits a field, or carries `gaps=[]`, must not erase
        what an earlier one established (§1b/§1c).
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
            if candidate.conclusion is not None:
                self._conclusion = candidate.conclusion
            for g in candidate.gaps or ():
                if g not in (self._gaps or ()):
                    self._gaps = [*(self._gaps or []), g]
            items = [(o.aspect, o) for o in candidate.observations]
        for key, finding in items:
            self.record(key, finding, evidence)

    def add_gap(
        self, gap: str, *, closes: str | None = None, establishes_kind: bool = False
    ) -> None:
        """Append a loop-authored caveat to the served envelope's `gaps`.

        `closes` names the aspect this gap accounts for (D4): a key that produced no
        grounded finding is still *addressed* as long as its failure is stated. Recording
        it here rather than inferring it later is what lets `addressed` be reconciled
        against real output instead of taken on trust.

        `establishes_kind` lets the loop's own step-8 gaps set `_kind` when the model never
        landed a single report. Without it a run where every search failed projects `None`
        — discarding the very gaps that explain *why* it failed, and falling back to raw
        excerpts as though nothing had gone wrong.
        """
        if gap not in (self._gaps or ()):
            self._gaps = [*(self._gaps or []), gap]
        if closes is not None:
            # Last-write-wins: a key closed more than once keeps its most recent reason.
            self._closed_as_gap[closes] = gap
        if establishes_kind and self._kind is None:
            self._kind = "analytical"

    def gap_reasons(self) -> dict[str, str]:
        """Keys closed via a stated gap, mapped to the reason text that closed them —
        the structured counterpart to `closed_as_gap()`'s bare key set, so a caller (e.g.
        retry logic) can join this against `AgentRunState.aspect_stats` instead of
        pattern-matching strings out of `_gaps`."""
        return dict(self._closed_as_gap)

    def closed_as_gap(self) -> set[str]:
        """Keys that produced no finding but did produce a stated gap (D4)."""
        return set(self._closed_as_gap)

    def keys(self) -> set[str]:
        return set(self._entries)

    def revised_keys(self) -> set[str]:
        """Keys whose finding was updated in place at least once (step 9).

        Tracked, but never a kill criterion: a run that concludes each aspect once,
        correctly, is a success. A revision is only expected when later evidence
        contradicts an earlier conclusion.
        """
        return {k for k, e in self._entries.items() if e.revisions > 0}

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
