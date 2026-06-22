from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
from dataclasses import dataclass
from typing import Literal

import httpx

from src.observability import langfuse as lf_client
from src.schemas.agent_findings import AgentFindings, AnalyticalFindings, EntityFinding

logger = logging.getLogger(__name__)

_FRANKFURTER_BASE = "https://api.frankfurter.dev/v1"
_FX_TIMEOUT = httpx.Timeout(3.0)

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_YEAR_RE = re.compile(r"^\d{4}$")


def _normalize_date(date: str | None) -> str | None:
    if not date:
        return None
    if _ISO_DATE_RE.match(date):
        return date
    if _YEAR_RE.match(date):
        return f"{date}-12-31"
    logger.warning("period_end_not_iso: %s — falling back to latest rate", date)
    return None  # frankfurter interprets None as "latest"


@dataclass(frozen=True, slots=True)
class NormalizedFinding:
    finding: EntityFinding
    normalized_value: (
        float | None
    )  # in target_currency; equals finding.value if no conversion needed
    fx_rate: float | None  # rate applied; None if same currency or no conversion


@dataclass(frozen=True, slots=True)
class ProcessedFindings:
    findings: tuple[NormalizedFinding, ...]
    answer_entity: str | None
    fx_rates_used: dict[str, float]  # key: "USD->EUR@2023-12-31"
    currency_converted: bool
    answer_note: str | None
    # Metadata carried for the renderer
    metric_requested: str | None = None
    target_currency: str | None = None
    comparison_op: Literal["argmin", "argmax", "list", "none"] | None = None
    analytical_findings: AnalyticalFindings | None = None


_UNIT_TO_MILLIONS: dict[str | None, float] = {
    "B": 1_000.0,
    "M": 1.0,
    "K": 0.001,
    "": 0.000_001,  # absolute / units
    None: 1.0,  # assume millions when unspecified
}


def _to_millions(value: float, unit: str | None) -> float:
    """Scale value to millions for unit-safe comparison."""
    return value * _UNIT_TO_MILLIONS.get(unit, 1.0)


def _normalizer_enabled() -> bool:
    return os.getenv("CURRENCY_NORMALIZER_ENABLED", "true").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


async def _fetch_rate(
    client: httpx.AsyncClient,
    from_cur: str,
    to_cur: str,
    date: str | None,
) -> tuple[str, float | None]:
    date_str = date or "latest"
    key = f"{from_cur}->{to_cur}@{date_str}"
    try:
        r = await client.get(
            f"{_FRANKFURTER_BASE}/{date_str}", params={"from": from_cur, "to": to_cur}
        )
        r.raise_for_status()
        rate = r.json()["rates"].get(to_cur)
        return key, rate
    except Exception as exc:
        logger.warning("FX fetch failed %s: %s", key, exc)
        return key, None


async def process_findings(findings: AgentFindings | AnalyticalFindings) -> ProcessedFindings:
    if isinstance(findings, AnalyticalFindings):
        return ProcessedFindings(
            findings=(),
            answer_entity=None,
            fx_rates_used={},
            currency_converted=False,
            answer_note=None,
            analytical_findings=findings,
        )

    available = [f for f in findings.findings if f.available and f.value is not None]
    target = findings.target_currency

    # Deterministic FX normalization fires whenever the agent reported a target
    # currency — for extraction (report values in USD) and comparison (normalize
    # before argmin/argmax) alike. This is the only FX path.
    needs_fx = (
        target is not None
        and _normalizer_enabled()
        and any(f.currency and f.currency != target for f in available)
    )

    fx_rates_used: dict[str, float] = {}
    currency_converted = False
    answer_note: str | None = None

    if needs_fx:
        assert target is not None  # narrowed above
        # Unique (from_currency, date) pairs requiring conversion
        pairs: list[tuple[str, str | None]] = list(
            {
                (f.currency, _normalize_date(f.period_end))
                for f in available
                if f.currency and f.currency != target
            }
        )

        lf = lf_client.get_client()
        _fx_lf_stack = contextlib.ExitStack()
        if lf:
            _fx_lf_stack.enter_context(
                lf.start_as_current_observation(
                    as_type="span",
                    name="fx_conversion",
                    input={"pairs": list(pairs), "target_currency": target},
                )
            )
        try:
            async with httpx.AsyncClient(timeout=_FX_TIMEOUT) as client:
                results = await asyncio.gather(
                    *[_fetch_rate(client, cur, target, date) for cur, date in pairs]
                )

            rate_map: dict[tuple[str, str | None], float | None] = {}
            failed: list[str] = []
            for (cur, date), (key, rate) in zip(pairs, results, strict=False):
                rate_map[(cur, date)] = rate
                if rate is not None:
                    fx_rates_used[key] = rate
                else:
                    failed.append(key)

            if lf:
                if failed:
                    lf.update_current_span(
                        level="ERROR",
                        status_message=f"FX fetch failed for: {', '.join(failed)}",
                        output={
                            "pairs_fetched": len(results),
                            "rates_ok": dict(fx_rates_used),
                            "rates_failed": failed,
                        },
                    )
                else:
                    lf.update_current_span(
                        output={
                            "pairs_fetched": len(results),
                            "rates_ok": fx_rates_used,
                        }
                    )
        finally:
            _fx_lf_stack.close()

        if failed:
            return ProcessedFindings(
                findings=tuple(
                    NormalizedFinding(finding=f, normalized_value=None, fx_rate=None)
                    for f in findings.findings
                ),
                answer_entity=None,
                fx_rates_used=fx_rates_used,
                currency_converted=False,
                answer_note=f"comparison not possible — FX conversion failed for: {', '.join(failed)}",
                metric_requested=findings.metric_requested,
                target_currency=target,
                comparison_op=findings.comparison_op,
            )

        normalized: list[NormalizedFinding] = []
        for f in findings.findings:
            if not f.available or f.value is None:
                normalized.append(NormalizedFinding(finding=f, normalized_value=None, fx_rate=None))
            elif f.currency and f.currency != target:
                rate = rate_map[(f.currency, _normalize_date(f.period_end))]
                norm_val = f.value * rate if rate is not None else None
                normalized.append(
                    NormalizedFinding(finding=f, normalized_value=norm_val, fx_rate=rate)
                )
            else:
                normalized.append(
                    NormalizedFinding(finding=f, normalized_value=f.value, fx_rate=None)
                )

        currency_converted = True

    else:
        normalized = [
            NormalizedFinding(
                finding=f,
                normalized_value=f.value if (f.available and f.value is not None) else None,
                fx_rate=None,
            )
            for f in findings.findings
        ]

    # Apply comparison op over available normalized values
    answer_entity: str | None = None
    op = findings.comparison_op
    if op in ("argmin", "argmax"):
        candidates = [n for n in normalized if n.normalized_value is not None]
        if candidates:

            def key_fn(n: NormalizedFinding) -> float:
                return _to_millions(n.normalized_value, n.finding.unit)  # type: ignore[arg-type]

            best = min(candidates, key=key_fn) if op == "argmin" else max(candidates, key=key_fn)
            answer_entity = best.finding.entity

    if len(available) == 1 and len(findings.findings) > 1:
        answer_note = "only one entity had available data"

    return ProcessedFindings(
        findings=tuple(normalized),
        answer_entity=answer_entity,
        fx_rates_used=fx_rates_used,
        currency_converted=currency_converted,
        answer_note=answer_note,
        metric_requested=findings.metric_requested,
        target_currency=target,
        comparison_op=findings.comparison_op,
    )


def _map_refs(raw_refs: list[str], chunk_id_to_ref: dict[str, str]) -> str:
    """Map chunk UUIDs to S-labels, silently dropping any without a context excerpt."""
    mapped = [chunk_id_to_ref[c] for c in raw_refs if c in chunk_id_to_ref]
    return ", ".join(mapped) or "—"


def _render_findings_block(
    processed: ProcessedFindings,
    chunk_id_to_ref: dict[str, str] | None = None,
) -> str:
    lines = ["[STRUCTURED FINDINGS]"]

    header_parts = []
    if processed.metric_requested:
        header_parts.append(f"Metric: {processed.metric_requested}")
    if processed.target_currency:
        header_parts.append(f"Target currency: {processed.target_currency}")
    if processed.comparison_op and processed.comparison_op != "none":
        header_parts.append(f"Operation: {processed.comparison_op}")
    if header_parts:
        lines.append(" | ".join(header_parts))

    if processed.answer_entity:
        ans_nf = next(
            (n for n in processed.findings if n.finding.entity == processed.answer_entity), None
        )
        if ans_nf and ans_nf.normalized_value is not None:
            cur = processed.target_currency or ans_nf.finding.currency or ""
            unit_str = ans_nf.finding.unit if ans_nf.finding.unit is not None else "M"
            lines.append(
                f"Answer: {processed.answer_entity} ({cur} {ans_nf.normalized_value:,.1f}{unit_str})"
            )
        else:
            lines.append(f"Answer: {processed.answer_entity}")

    if processed.fx_rates_used:
        fx_parts = [f"{k}: {v:.4f}" for k, v in processed.fx_rates_used.items()]
        lines.append("FX rates used: " + " | ".join(fx_parts))

    if processed.answer_note:
        lines.append(f"Note: {processed.answer_note}")

    lines.append("")

    for nf in processed.findings:
        f = nf.finding
        unit_str = f.unit if f.unit is not None else "M"
        raw_chunks = f.source_chunks or []
        if chunk_id_to_ref is not None:
            # Drop refs with no excerpt in the synthesis context — leaking a raw ref
            # here would let the model cite an ID the citation pipeline can't resolve.
            chunks_str = _map_refs(raw_chunks, chunk_id_to_ref)
        else:
            chunks_str = ", ".join(raw_chunks) or "—"
        if not f.available or f.value is None:
            reason = f.reason or "not found in retrieved context"
            lines.append(f"{f.entity:<22} | N/A | not available: {reason}")
        elif nf.fx_rate is not None and nf.normalized_value is not None:
            to_cur = processed.target_currency or ""
            native = f"{f.currency} {f.value:,.1f}{unit_str}"
            converted = f"{to_cur} {nf.normalized_value:,.1f}{unit_str}"
            lines.append(
                f"{f.entity:<22} | {converted:<14} | from {native:<16} | rate: {nf.fx_rate:.4f}"
                f" | period: {f.period_end or '—'} | chunks: {chunks_str}"
            )
        else:
            cur = f.currency or ""
            val_str = f"{cur} {f.value:,.1f}{unit_str}" if cur else f"{f.value:,.1f}{unit_str}"
            lines.append(
                f"{f.entity:<22} | {val_str:<14} | native"
                f" | period: {f.period_end or '—'} | chunks: {chunks_str}"
            )

    lines.append("[END STRUCTURED FINDINGS]")
    return "\n".join(lines)


def _render_observations_block(
    findings: AnalyticalFindings,
    chunk_id_to_ref: dict[str, str] | None = None,
) -> str:
    lines = ["[AGENT OBSERVATIONS]", f"Question: {findings.question}", ""]

    for i, obs in enumerate(findings.observations, 1):
        if chunk_id_to_ref is not None:
            chunks_str = _map_refs(obs.evidence_chunks, chunk_id_to_ref)
            refuted_str = _map_refs(obs.refuted_by or [], chunk_id_to_ref)
        else:
            chunks_str = ", ".join(obs.evidence_chunks) if obs.evidence_chunks else "—"
            refuted_str = ", ".join(obs.refuted_by) if obs.refuted_by else "—"
        lines.append(
            f"{i}. [{obs.confidence} confidence] {obs.claim}"
            f" | evidence: {chunks_str} | refuted_by: {refuted_str}"
        )

    if findings.conclusion:
        lines.append(f"\nConclusion: {findings.conclusion}")

    if findings.gaps:
        lines.append("Gaps: " + "; ".join(findings.gaps))

    lines.append("[END AGENT OBSERVATIONS]")
    return "\n".join(lines)
