"""Text and status for the agent timeline. Pure functions, no I/O.

Every label, note and outcome the timeline shows is decided here from structured fields:
copied verbatim or filled into a fixed template, never paraphrased from model output.
`loop.py` and `tasks.py` call these and emit the result as `activity` events; the UI
renders them as they arrive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, NamedTuple
from uuid import UUID

from pydantic import ValidationError

from src.schemas.agent_findings import Finding
from src.schemas.query_router import DocumentScopeResult, RouterOutput
from src.services.chat.agent.findings import FLAGGED_MARKER, RecordOutcome
from src.services.chat.agent.number_grounding import strip_marks
from src.services.chat.agent.tools import SearchDocumentsArgs
from src.services.chat.events import ActivityStatus
from src.utils.config import get_agent_activity_max_sources

if TYPE_CHECKING:
    from src.services.chat.agent.loop import _SearchResult
    from src.services.chat.agent.state import AgentLoopMeta

# Stop reasons where the run ended without deciding to: the answer may be incomplete.
WARN_STOPS = frozenset({"timeout", "deadline", "budget_cap", "llm_error", "truncated"})

NOT_RUN_FINAL_PASS = "not run — final pass records findings only"

# Per document a search showed the model, in first-shown order: (name, sorted pages).
SearchSources = list[tuple[str, list[int]]]

_CAPABILITY_TEXT = {"dense": "semantic search", "keyword": "keyword search", "rerank": "reranking"}

# How one finding in a report fared: the ledger's outcome, or rejected by `_apply_report`
# before reaching it.
ReportOutcome = RecordOutcome | Literal["unknown", "unsearched"]

_NOT_RECORDED = {
    "unknown": "Note not recorded — wrong key, retried",
    "unsearched": "Note not recorded — not searched yet",
    "uncited": "Note not recorded — no supporting excerpt",
    "blocked": "Note not recorded — blocked by safety check",
}


class NoteRow(NamedTuple):
    status: ActivityStatus
    label: str
    note: str | None = None
    claim: str | None = None  # the stored claim, shown on hover


MALFORMED_REPORT_ROW = NoteRow("warn", "Note not recorded — malformed")


@dataclass(frozen=True)
class Outcome:
    """The pill and the closing lines of the expanded view; `detail` of the `outcome` event.

    ``hidden`` drops the pill (a clarification card is the response)."""

    status: ActivityStatus
    text: str
    lines: list[str] = field(default_factory=list)
    hidden: bool = False

    def to_detail(self) -> dict:
        return {
            "status": self.status,
            "text": self.text,
            "lines": list(self.lines),
            "hidden": self.hidden,
        }


@dataclass(frozen=True)
class StageResult:
    """How a stage went; attached to its `stage_ended`. ``hidden=None`` keeps the stage's
    default visibility."""

    status: ActivityStatus = "ok"
    note: str | None = None
    hidden: bool | None = None
    companies: list[dict] | None = None

    def to_detail(self, default_hidden: bool) -> dict:
        detail: dict = {
            "status": self.status,
            "hidden": default_hidden if self.hidden is None else self.hidden,
        }
        if self.note is not None:
            detail["note"] = self.note
        if self.companies is not None:
            detail["companies"] = self.companies
        return detail


STAGE_FAILED = StageResult("fail", hidden=False)


def _count(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def _found(scope: DocumentScopeResult) -> list[str]:
    return [name for name, ids in (scope.per_entity_doc_ids or {}).items() if ids]


def _years(scope: DocumentScopeResult, name: str) -> list[int]:
    for item in scope.entity_manifest or []:
        if item.entity_name == name:
            return sorted({d["year"] for d in item.doc_summaries if d.get("year")}, reverse=True)
    return []


def _year_text(years: list[int]) -> str:
    return ", ".join(str(y) for y in years)


def where(scope: DocumentScopeResult | None) -> str:
    """Where the run looked: the companies in scope, or the documents when none is named."""
    if scope is None:
        return "your documents"
    found = _found(scope)
    if len(found) == 1:
        years = _years(scope, found[0])
        return f"{found[0]} ({_year_text(years)})" if years else found[0]
    if len(found) == 2:
        return f"{found[0]} and {found[1]}"
    if found:
        return f"{len(found)} companies"
    if scope.doc_ids is None:
        return "your documents"
    n = len(scope.doc_ids)
    if scope.source == "explicit":
        return _count(n, "selected document", "selected documents")
    if scope.source == "filtered":
        return _count(n, "filtered document", "filtered documents")
    if scope.source == "all":
        return f"all {_count(n, 'document', 'documents')}"
    return _count(n, "document", "documents")


def route_stage(router_output: RouterOutput, scope: DocumentScopeResult | None) -> StageResult:
    """`Understood question` row: what the router decided to look at."""
    if router_output.route == "out_of_scope":
        return StageResult()
    if router_output.route != "retrieval":
        return StageResult(note="answer without searching")
    shape = router_output.query_shape
    if shape == "comparison" and scope is not None and scope.per_entity_doc_ids:
        companies = [
            {"name": name, "years": _years(scope, name), "found": bool(ids)}
            for name, ids in scope.per_entity_doc_ids.items()
        ]
        return StageResult(note=_count(len(companies), "company", "companies"), companies=companies)
    if shape == "analytical":
        return StageResult(note=f"explain · {where(scope)}")
    found = _found(scope) if scope is not None else []
    if len(found) == 1 and scope is not None:
        years = _years(scope, found[0])
        return StageResult(note=f"{found[0]} · {_year_text(years)}" if years else found[0])
    return StageResult(note=where(scope))


def agent_stage(meta: AgentLoopMeta) -> StageResult:
    """`Searched documents` row: search and pass counts, and how the run ended."""
    searches = meta.tool_calls_total - meta.report_calls_total
    note = f"{_count(searches, 'search', 'searches')}, {_count(meta.iterations, 'pass', 'passes')}"
    if meta.convergence_reason == "search_unavailable":
        return StageResult("fail", note)
    if not meta.sealed or meta.convergence_reason in WARN_STOPS or meta.degraded_capabilities:
        return StageResult("warn", note)
    return StageResult("ok", note)


def scan_stage(severity: str) -> StageResult | None:
    """`Checked input` row, shown only when the scan changed or stopped something."""
    if severity == "block":
        return StageResult("fail", hidden=False)
    if severity == "flag":
        return StageResult("warn", "removed suspicious instructions", hidden=False)
    return None


def search_label(arguments: str, per_entity_doc_ids: dict[str, list[UUID]] | None) -> str:
    """A search row's label: the model's query in quotes, prefixed with the company it
    names when more than one company is in scope."""
    try:
        args = SearchDocumentsArgs.model_validate_json(arguments)
    except ValidationError:
        return "malformed search"
    in_scope = [name for name, ids in (per_entity_doc_ids or {}).items() if ids]
    if args.entity and len(in_scope) > 1:
        return f'{args.entity}: "{args.query}"'
    return f'"{args.query}"'


def search_result_row(
    result: _SearchResult, new: int, timeout_s: float
) -> tuple[ActivityStatus, str]:
    """A finished search row's status and note. First match wins: timed out, backend
    failed, not in library, malformed, then what came back (degraded marks it warn)."""
    if result.timed_out:
        return "warn", f"timed out after {timeout_s:g}s"
    if result.backend_failed:
        return "warn", "search failed"
    if result.not_found:
        return "skipped", f"{result.entity} is not in your documents"
    if result.args_invalid:
        return "skipped", "malformed search, skipped"

    if new:
        status: ActivityStatus = "ok"
        note = f"+{new} new"
    elif result.chunks:
        status, note = "skipped", "nothing new"
    else:
        status, note = "skipped", "no results"

    if result.degraded:
        return "warn", f"{note} · {_capabilities(result.degraded)} unavailable"
    return status, note


def search_sources(sources: SearchSources) -> str | None:
    """Where a search's excerpts came from, for the row's tooltip; None when it showed none."""
    shown = [
        f"{name} {', '.join(f'p.{p}' for p in pages)}" if pages else name
        for name, pages in sources[: get_agent_activity_max_sources()]
    ]
    return " · ".join(shown) or None


def _capabilities(caps: frozenset[str]) -> str:
    return " and ".join(_CAPABILITY_TEXT[c] for c in sorted(caps))


def report_rows(
    outcomes: list[tuple[ReportOutcome, Finding | None]], plan: dict[str, str]
) -> list[NoteRow]:
    """One note row per finding in a report, labelled by the plan item it covers. The claim
    rides along only as the ledger stored it (after the injection screen), never the raw
    tool argument."""
    rows: list[NoteRow] = []
    for outcome, stored in outcomes:
        if outcome != "accepted" or stored is None:
            rows.append(NoteRow("warn", _NOT_RECORDED[outcome]))
            continue
        claim = strip_marks(stored.claim)
        cleaned = claim.startswith(FLAGGED_MARKER)
        if cleaned:
            claim = claim.removeprefix(FLAGGED_MARKER)
        status, prefix = ("ok", "Noted") if stored.supported else ("skipped", "Not in documents")
        rows.append(
            NoteRow(
                status,
                f"{prefix}: {plan.get(stored.key, stored.key)}",
                "text cleaned" if cleaned else None,
                claim,
            )
        )
    return rows


# How a request ended. A direct answer's follow-up reuse is told by `carried_outcome`.
ExitPath = Literal[
    "agent", "direct", "clarification", "too_broad", "out_of_scope", "blocked", "error"
]

_STOP_TEXT = {
    "covered": "everything planned was covered",
    "convergence": "recent searches found nothing new",
    "iteration_cap": "reached the search limit",
    "budget_cap": "reached the cost limit",
    "timeout": "the search model took too long",
    "deadline": "ran out of time",
    "llm_error": "the search model failed",
    "truncated": "the search model's reply was cut off",
    "search_unavailable": "document search was unavailable",
}

_FALLBACK_PILL = {
    "no_findings": "no findings recorded, answered from excerpts",
    "no_citations": "no finding cited an excerpt, answered from excerpts",
}
_FALLBACK_LINE = {
    "no_findings": "No findings were recorded; the answer is drawn directly from retrieved "
    "excerpts.",
    "no_citations": "No finding cited an excerpt; the answer is drawn directly from retrieved "
    "excerpts.",
}

_CARRIED = {
    "carried": Outcome("ok", "Reused findings from your previous question"),
    "dropped_scope": Outcome(
        "warn", "Earlier findings don't match the current scope — answered without documents"
    ),
    "dropped_hop_cap": Outcome(
        "warn", "Earlier findings too far back to reuse — answered without documents"
    ),
}


@dataclass(frozen=True)
class OutcomeInputs:
    exit_path: ExitPath
    scope: DocumentScopeResult | None = None
    query_shape: str | None = None
    meta: AgentLoopMeta | None = None
    unresolved: tuple[str, ...] = ()
    synthesis_fallback: str | None = None
    carried_outcome: str | None = None
    too_broad_count: int | None = None
    max_companies: int | None = None


def stop_text(meta: AgentLoopMeta) -> str:
    if meta.convergence_reason == "natural":
        return "finished" if meta.sealed else "the search model stopped before covering everything"
    return _STOP_TEXT[meta.convergence_reason]


def build_outcome(inputs: OutcomeInputs) -> Outcome:
    """The pill and closing lines, once per request. Highest match wins the pill; the
    agent path's other conditions go to the lines."""
    path = inputs.exit_path
    if path == "error":
        return Outcome("fail", "Something went wrong")
    if path == "blocked":
        return Outcome("fail", "Request blocked by the safety check")
    if path == "out_of_scope":
        return Outcome("skipped", "Not answered: outside what this assistant covers")
    if path == "clarification":
        return Outcome("ok", "", hidden=True)
    if path == "too_broad":
        return Outcome(
            "skipped",
            f"Not run: the question covers {inputs.too_broad_count} companies; "
            f"up to {inputs.max_companies} per question",
        )
    if path == "direct" or inputs.meta is None:
        return _CARRIED.get(
            inputs.carried_outcome or "", Outcome("ok", "Answered without searching documents")
        )
    return _agent_outcome(inputs, inputs.meta)


def _agent_outcome(inputs: OutcomeInputs, meta: AgentLoopMeta) -> Outcome:
    reason = meta.convergence_reason
    per_entity = (inputs.scope.per_entity_doc_ids or {}) if inputs.scope else {}
    missing = [name for name, ids in per_entity.items() if not ids]
    missing_text = missing[0] if len(missing) == 1 else f"{len(missing)} companies"
    names_missing = bool(missing) and inputs.query_shape in ("extraction", "comparison")
    searched = f"Searched {where(inputs.scope)}"
    searches = meta.tool_calls_total - meta.report_calls_total

    lines = [f"Stopped: {stop_text(meta)}", *inputs.unresolved]
    if missing and not names_missing:
        lines.append(f"Not in your documents: {', '.join(missing)}")
    if meta.degraded_capabilities:
        lines.append(
            f"Some searches ran without {_capabilities(meta.degraded_capabilities)}; "
            "results may be less relevant."
        )
    if inputs.synthesis_fallback:
        lines.append(_FALLBACK_LINE[inputs.synthesis_fallback])

    warn = (
        not meta.sealed
        or bool(missing)
        or reason in WARN_STOPS
        or bool(meta.degraded_capabilities)
        or inputs.synthesis_fallback is not None
    )
    if reason == "search_unavailable":
        status: ActivityStatus = "fail"
        text = "Document search unavailable — answer not grounded"
    elif names_missing:
        status = "warn"
        found = len(per_entity) - len(missing)
        text = (
            f"Searched {found} of {len(per_entity)} companies — {missing_text} not in your "
            "documents"
        )
    elif reason in WARN_STOPS:
        status, text = "warn", f"Stopped early: {stop_text(meta)}"
    # Above "not sealed": a run with no usable findings rarely seals, and the fallback is
    # the more useful thing to say about it.
    elif inputs.synthesis_fallback:
        status = "warn"
        text = f"{searched} · {_FALLBACK_PILL[inputs.synthesis_fallback]}"
    elif not meta.sealed:
        status, text = "warn", f"Stopped early: {stop_text(meta)}"
    elif meta.degraded_capabilities:
        caps = _capabilities(meta.degraded_capabilities)
        status, text = "warn", f"{searched} · partial search ({caps} unavailable)"
    else:
        status = "warn" if warn else "ok"
        text = f"{searched} · {_count(searches, 'search', 'searches')}"
    return Outcome(status, text, lines)
