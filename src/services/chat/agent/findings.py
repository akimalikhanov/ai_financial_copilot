"""FindingsLedger — the run's keyed store of concluded findings (the scratchpad).

The third state store, beside `Transcript` (the model's view) and `EvidenceLedger`
(retrieved chunks). Where the transcript logs *interactions*, this stores *conclusions*,
addressed by a stable key — ``EntityFinding.entity`` or ``Observation.aspect`` — and
updated in place.

The loop folds in every `report_*` call as it arrives — reports are incremental, not
terminal — and projects the accumulation back for synthesis via `projection`.

`record` updates in place, so each key holds its latest finding and there is no best-of
comparator. A positive claim whose chunk refs don't resolve in the `EvidenceLedger` is
dropped, not admitted, and the prior entry for that key is left intact. This grounding
filter is the only correctness filter on what reaches synthesis.

Every model-written string is also scanned like a retrieved excerpt before it is stored,
because it reaches the answering model as findings, outside any excerpt tag: the tool
model can paraphrase an instruction it read in an excerpt into its own words.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal
from uuid import UUID

from src.schemas.agent_findings import (
    AgentFindings,
    AnalyticalFindings,
    EntityFinding,
    Observation,
)
from src.services.security.injection_detector import scan_retrieved_chunk

if TYPE_CHECKING:
    from src.services.chat.agent.evidence import EvidenceLedger

logger = logging.getLogger(__name__)

Candidate = AgentFindings | AnalyticalFindings

# Prefixed to model-written text the injection scan flags; the synthesis prompt names it.
FLAGGED_MARKER = "[flagged] "

_DEGRADED_CAVEAT = (
    "The search did not fully converge; these findings are partial and may be incomplete."
)


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


def _is_positive(finding: EntityFinding | Observation) -> bool:
    """A stated negative ("not available", `substantiated: false`) is a real conclusion
    about its key and cites nothing by definition, so only positive claims need grounding."""
    return finding.available if isinstance(finding, EntityFinding) else finding.substantiated


def _citations(finding: EntityFinding | Observation) -> list[str]:
    return finding.source_chunks if isinstance(finding, EntityFinding) else finding.evidence_chunks


def finding_chunk_ids(findings: Candidate) -> set[str]:
    """All chunk-id strings referenced by the findings."""
    ids: set[str] = set()
    if isinstance(findings, AgentFindings):
        for f in findings.findings:
            ids.update(f.source_chunks or [])
    else:
        for o in findings.observations:
            ids.update(o.evidence_chunks or [])
    return ids


class FindingsLedger:
    def __init__(self) -> None:
        self._entries: dict[str, EntityFinding | Observation] = {}
        # Set by the first `ingest` or `record`; projection serves None until then.
        self._reported = False
        # Report-envelope metadata, last-write-wins — the per-item entries alone cannot
        # reconstruct the AgentFindings/AnalyticalFindings shape synthesis and persistence
        # expect, so the envelope is retained here rather than re-derived.
        self._metric_requested: str | None = None
        self._comparison_op: Literal["argmin", "argmax", "list", "none"] | None = None
        self._question: str | None = None
        self._conclusion: str | None = None
        self._gaps: list[str] | None = None
        # Positive claims seen, and those dropped because none of their citations resolve.
        self._claims = 0
        self._uncited = 0
        # Model-written strings the injection scan flagged or blocked.
        self._screened = {"flag": 0, "block": 0}

    def _screen(self, text: str | None) -> str | None:
        """`text` scanned with the excerpt thresholds: None when blocked, the sanitized text
        behind `FLAGGED_MARKER` when flagged, unchanged when clean."""
        if not text:
            return text
        signal = scan_retrieved_chunk(text)
        if signal.severity == "clean":
            return text
        self._screened[signal.severity] += 1
        logger.warning(
            "agent_finding_injection",
            extra={"severity": signal.severity, "matched_rules": signal.matched_rules},
        )
        return None if signal.severity == "block" else FLAGGED_MARKER + signal.sanitized_text

    def _screen_finding(
        self, finding: EntityFinding | Observation
    ) -> EntityFinding | Observation | None:
        field = "claim" if isinstance(finding, Observation) else "reason"
        text = getattr(finding, field)
        screened = self._screen(text)
        if text and screened is None:
            return None
        return finding if screened == text else finding.model_copy(update={field: screened})

    def record(
        self, key: str, finding: EntityFinding | Observation, evidence: EvidenceLedger
    ) -> bool:
        """Insert or update-in-place. Returns False (and leaves any prior entry intact)
        when the injection scan blocks the item's text or the grounding filter drops it."""
        self._reported = True
        screened = self._screen_finding(finding)
        if screened is None:
            return False
        finding = screened
        if _is_positive(finding):
            self._claims += 1
            if not _resolves(_citations(finding), evidence):
                self._uncited += 1
                return False
        self._entries[key] = finding
        return True

    def ingest(
        self, candidate: AgentFindings | AnalyticalFindings, evidence: EvidenceLedger
    ) -> None:
        """Fold one report into the ledger. Accumulates; never prunes.

        Restated keys update in place; keys this report omits are left alone. Reports are
        incremental rather than a single terminal restatement, so omission carries no
        information at all — the model reports an aspect when its evidence settles and
        never restates the others.

        Envelope fields are last-write-wins but null-guarded, and `gaps` unions rather than
        replaces: a later report that omits a field, or carries `gaps=[]`, must not erase
        what an earlier one established. A field the injection scan blocks counts as
        omitted.

        One run only ever offers one report tool (the loop answers any other with "tool
        not available"), so every candidate a ledger sees has the same type.
        """
        self._reported = True
        items: list[tuple[str, EntityFinding | Observation]]
        if isinstance(candidate, AgentFindings):
            self._metric_requested = (
                self._screen(candidate.metric_requested) or self._metric_requested
            )
            self._comparison_op = candidate.comparison_op or self._comparison_op
            items = [(f.entity, f) for f in candidate.findings]
        else:
            self._question = self._screen(candidate.question) or self._question
            conclusion = self._screen(candidate.conclusion)
            if conclusion is not None:
                self._conclusion = conclusion
            for gap in candidate.gaps or ():
                g = self._screen(gap)
                if g is not None and g not in (self._gaps or ()):
                    self._gaps = [*(self._gaps or []), g]
            items = [(o.aspect, o) for o in candidate.observations]
        for key, finding in items:
            self.record(key, finding, evidence)

    def keys(self) -> set[str]:
        return set(self._entries)

    def uncited_claim_rate(self) -> float:
        """Share of positive claims `record` dropped because no citation resolved. Stated
        negatives cite nothing by design and are not claims here."""
        return self._uncited / self._claims if self._claims else 0.0

    def get(self, key: str) -> EntityFinding | Observation | None:
        return self._entries.get(key)

    def screened(self) -> dict[str, int]:
        """Model-written strings the injection scan flagged or blocked so far."""
        return dict(self._screened)

    def projection(
        self, *, analytical: bool, degraded: bool = False, unresolved: Sequence[str] = ()
    ) -> AgentFindings | AnalyticalFindings | None:
        """The findings synthesis serves, in the run's shape. None when no report was ever
        attempted (raw-excerpt fallback); otherwise the accumulated ledger reconstructed
        into its report shape, marked degraded when the run never sealed. Degraded only
        annotates the analytical path, whose `gaps` field can carry the caveat.

        `unresolved` holds one line per plan key still open, appended to `gaps`, so an
        analytical run where no report ever landed still serves the lines that explain
        why instead of falling back to raw excerpts.
        """
        if not analytical:
            if not self._reported:
                return None
            findings = tuple(f for f in self._entries.values() if isinstance(f, EntityFinding))
            return AgentFindings(
                metric_requested=self._metric_requested or "",
                findings=findings,
                comparison_op=self._comparison_op,
            )
        if not self._reported and not unresolved:
            return None
        observations = tuple(f for f in self._entries.values() if isinstance(f, Observation))
        gaps = list(self._gaps) if self._gaps else []
        gaps.extend(g for g in unresolved if g not in gaps)
        if degraded:
            gaps.append(_DEGRADED_CAVEAT)
        return AnalyticalFindings(
            question=self._question or "",
            observations=observations,
            conclusion=self._conclusion,
            gaps=gaps or None,
        )
