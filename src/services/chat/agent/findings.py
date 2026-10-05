"""FindingsLedger — the run's keyed store of concluded findings (the scratchpad).

The third state store, beside `Transcript` (the model's view) and `EvidenceLedger`
(retrieved chunks). Where the transcript logs *interactions*, this stores *conclusions*,
addressed by their plan key (an entity name or an aspect id) and updated in place.

The loop folds in every `report_findings` call as it arrives — reports are incremental,
not terminal — and projects the accumulation back for synthesis via `projection`.

`record` updates in place, so each key holds its latest finding and there is no best-of
comparator. The exception is figures: a supported finding restated with figures for
another metric or period adds them to the ones already held, so a two-period question
keeps both periods. A positive claim whose chunk refs don't resolve in the
`EvidenceLedger` is dropped, not admitted, and the prior entry for that key is left
intact. This grounding filter is the only correctness filter on what reaches synthesis.

Every model-written string is also scanned like a retrieved excerpt before it is stored,
because it reaches the answering model as findings, outside any excerpt tag: the tool
model can paraphrase an instruction it read in an excerpt into its own words.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal
from uuid import UUID

from src.schemas.agent_findings import AgentFindings, Figure, Finding, FindingsReport
from src.services.security.injection_detector import scan_retrieved_chunk

if TYPE_CHECKING:
    from src.services.chat.agent.evidence import EvidenceLedger

logger = logging.getLogger(__name__)

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


def _figure_key(figure: Figure) -> tuple[str, str | None]:
    return figure.metric.strip().casefold(), figure.period_end or figure.fiscal_label


def _merge(prior: Finding, new: Finding) -> Finding:
    """`new` with the prior figures it doesn't restate, and the evidence they cite."""
    figures = {_figure_key(f): f for f in prior.figures}
    figures.update((_figure_key(f), f) for f in new.figures)
    evidence = list(dict.fromkeys([*prior.evidence, *new.evidence]))
    return new.model_copy(update={"figures": list(figures.values()), "evidence": evidence})


class FindingsLedger:
    def __init__(self) -> None:
        self._entries: dict[str, Finding] = {}
        # Set by the first `ingest` or `record`; projection serves None until then.
        self._reported = False
        # Report-envelope fields, last write wins.
        self._comparison_op: Literal["argmin", "argmax", "list", "none"] | None = None
        self._conclusion: str | None = None
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

    def _screen_finding(self, finding: Finding) -> Finding | None:
        """The finding with its free-text fields screened; None if any is blocked."""
        claim = self._screen(finding.claim)
        if claim is None:
            return None
        figures: list[Figure] = []
        for fig in finding.figures:
            metric, label = self._screen(fig.metric), self._screen(fig.fiscal_label)
            if metric is None or (fig.fiscal_label and label is None):
                return None
            figures.append(fig.model_copy(update={"metric": metric, "fiscal_label": label}))
        return finding.model_copy(update={"claim": claim, "figures": figures})

    def record(self, key: str, finding: Finding, evidence: EvidenceLedger) -> bool:
        """Insert or update-in-place. Returns False (and leaves any prior entry intact)
        when the injection scan blocks the item's text or the grounding filter drops it."""
        self._reported = True
        screened = self._screen_finding(finding)
        if screened is None:
            return False
        finding = screened
        if finding.supported:
            self._claims += 1
            if not _resolves(finding.evidence, evidence):
                self._uncited += 1
                return False
            prior = self._entries.get(key)
            if prior is not None and prior.supported:
                finding = _merge(prior, finding)
        self._entries[key] = finding
        return True

    def ingest(self, report: FindingsReport, evidence: EvidenceLedger) -> None:
        """Fold one report into the ledger. Accumulates; never prunes.

        Restated keys update in place; keys this report omits are left alone. Reports are
        incremental rather than a single terminal restatement, so omission carries no
        information at all — the model reports a key when its evidence settles and never
        restates the others.

        Envelope fields are last-write-wins but null-guarded: a later report that omits a
        field must not erase what an earlier one established. A field the injection scan
        blocks counts as omitted.
        """
        self._reported = True
        self._comparison_op = report.comparison_op or self._comparison_op
        self._conclusion = self._screen(report.conclusion) or self._conclusion
        for finding in report.findings:
            self.record(finding.key, finding, evidence)

    def keys(self) -> set[str]:
        return set(self._entries)

    def uncited_claim_rate(self) -> float:
        """Share of positive claims `record` dropped because no citation resolved. Stated
        negatives cite nothing by design and are not claims here."""
        return self._uncited / self._claims if self._claims else 0.0

    def get(self, key: str) -> Finding | None:
        return self._entries.get(key)

    def screened(self) -> dict[str, int]:
        """Model-written strings the injection scan flagged or blocked so far."""
        return dict(self._screened)

    def projection(
        self, *, degraded: bool = False, unresolved: Sequence[str] = ()
    ) -> AgentFindings | None:
        """The findings synthesis serves. None when no report was ever attempted and no
        key is left open (raw-excerpt fallback).

        `unresolved` holds one line per plan key still open, so a run where no report
        ever landed still serves the lines that explain why. A run that never sealed also
        carries the degraded caveat.
        """
        if not self._reported and not unresolved:
            return None
        lines = [*unresolved, *([_DEGRADED_CAVEAT] if degraded else [])]
        return AgentFindings(
            findings=tuple(self._entries.values()),
            comparison_op=self._comparison_op,
            conclusion=self._conclusion,
            unresolved=tuple(lines),
        )
