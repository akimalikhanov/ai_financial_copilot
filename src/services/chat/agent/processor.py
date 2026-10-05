"""FX normalization + findings rendering (moved from chat/findings_processor.py)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import random
import re
from dataclasses import dataclass
from dataclasses import replace as dc_replace
from uuid import UUID

import httpx

from src.observability.langfuse import span as lf_span
from src.observability.metrics import CITATION_REFS_DROPPED
from src.schemas.agent_findings import AgentFindings, Figure
from src.schemas.retrieval import RAGContext
from src.services.chat.agent.number_grounding import NumberGrounding, to_millions, verify_value

logger = logging.getLogger(__name__)

_FRANKFURTER_BASE = "https://api.frankfurter.dev/v1"
_FX_TIMEOUT = httpx.Timeout(3.0)

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _normalize_date(date: str | None) -> str | None:
    """`date` if it is ISO, else None (the latest rate). A bare year is not turned into
    Dec 31: that guess is wrong for every non-calendar fiscal year."""
    if not date:
        return None
    if _ISO_DATE_RE.match(date):
        return date
    logger.warning("period_end_not_iso: %s — falling back to latest rate", date)
    return None  # frankfurter interprets None as "latest"


@dataclass(frozen=True, slots=True)
class NormalizedFigure:
    key: str
    figure: Figure
    normalized_amount: float | None  # in target_currency; the amount if not converted
    fx_rate: float | None  # rate applied; None if same currency or no conversion
    number_grounding: NumberGrounding = NumberGrounding.UNVERIFIABLE


@dataclass(frozen=True, slots=True)
class ProcessedFindings:
    findings: AgentFindings
    # One row per figure of a supported finding, in finding order.
    figures: tuple[NormalizedFigure, ...]
    answer_entity: str | None
    fx_rates_used: dict[str, float]  # key: "USD->EUR@2023-12-31"
    currency_converted: bool
    answer_note: str | None
    target_currency: str | None = None

    def figures_for(self, key: str) -> list[NormalizedFigure]:
        return [n for n in self.figures if n.key == key]


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
    for attempt in range(2):
        try:
            r = await client.get(
                f"{_FRANKFURTER_BASE}/{date_str}", params={"from": from_cur, "to": to_cur}
            )
            r.raise_for_status()
            rate = r.json()["rates"].get(to_cur)
            return key, rate
        except Exception as exc:
            if attempt == 0:
                await asyncio.sleep(0.1 + random.uniform(0, 0.15))
                logger.debug("FX fetch retry %s: %s", key, exc)
            else:
                logger.warning("FX fetch failed %s: %s", key, exc)
    return key, None


_DEFAULT_COMPARISON_CURRENCY = "USD"


async def process_findings(
    findings: AgentFindings,
    requested_currency: str | None = None,
    chunk_texts: dict[str, str] | None = None,
) -> ProcessedFindings:
    """FX-normalize and rank the figures of a run's supported findings. A run without
    figures passes through unchanged, with no FX call.

    `chunk_texts` (chunk-UUID string -> sanitized rendered text) enables the
    number-grounding check: does the cited excerpt actually contain the asserted value?
    Omitted (the default), every figure's `number_grounding` stays `UNVERIFIABLE` — this
    is purely additive instrumentation, never a filter (see `number_grounding.py`).
    """
    rows = [(f.key, fig) for f in findings.findings if f.supported for fig in f.figures]
    available = [fig for _, fig in rows]
    op = findings.comparison_op
    is_comparison = op in ("argmin", "argmax")

    # Currency resolution — pure code, no LLM decision:
    # 1. If the user named a currency in the query, use it.
    # 2. If findings span multiple currencies and we need to rank, fall back to the
    #    product default and disclose it in answer_note.
    # 3. Otherwise no conversion target — render native values.
    currencies = {f.currency for f in available if f.currency}
    multi_ccy = len(currencies) > 1
    answer_note: str | None = None

    if requested_currency:
        resolved_target: str | None = requested_currency
    elif multi_ccy and is_comparison:
        resolved_target = _DEFAULT_COMPARISON_CURRENCY
        answer_note = (
            f"no target currency specified — compared in {resolved_target} "
            f"(findings span {', '.join(sorted(currencies))})"
        )
    else:
        resolved_target = None

    needs_fx = (
        resolved_target is not None
        and _normalizer_enabled()
        and any(f.currency and f.currency != resolved_target for f in available)
    )

    fx_rates_used: dict[str, float] = {}
    currency_converted = False

    if needs_fx:
        if resolved_target is None:  # narrowed above; keep the invariant under -O
            raise RuntimeError("needs_fx implies a resolved target currency")
        # Unique (from_currency, date) pairs requiring conversion
        pairs: list[tuple[str, str | None]] = list(
            {
                (f.currency, _normalize_date(f.period_end))
                for f in available
                if f.currency and f.currency != resolved_target
            }
        )

        with lf_span(
            "fx_conversion", input={"pairs": list(pairs), "target_currency": resolved_target}
        ) as obs:
            async with httpx.AsyncClient(timeout=_FX_TIMEOUT) as client:
                results = await asyncio.gather(
                    *[_fetch_rate(client, cur, resolved_target, date) for cur, date in pairs]
                )

            rate_map: dict[tuple[str, str | None], float | None] = {}
            failed: list[str] = []
            for (cur, date), (key, rate) in zip(pairs, results, strict=False):
                rate_map[(cur, date)] = rate
                if rate is not None:
                    fx_rates_used[key] = rate
                else:
                    failed.append(key)

            if obs:
                if failed:
                    obs.update(
                        level="ERROR",
                        status_message=f"FX fetch failed for: {', '.join(failed)}",
                        output={
                            "pairs_fetched": len(results),
                            "rates_ok": dict(fx_rates_used),
                            "rates_failed": failed,
                        },
                    )
                else:
                    obs.update(
                        output={
                            "pairs_fetched": len(results),
                            "rates_ok": fx_rates_used,
                        }
                    )

        if failed:
            # For argmin/argmax we can't rank with a hole — abort the whole result.
            # For list/none, keep what converted; failed pairs stay in native currency.
            if is_comparison:
                return ProcessedFindings(
                    findings=findings,
                    figures=tuple(
                        NormalizedFigure(key=k, figure=fig, normalized_amount=None, fx_rate=None)
                        for k, fig in rows
                    ),
                    answer_entity=None,
                    fx_rates_used=fx_rates_used,
                    currency_converted=False,
                    answer_note=f"comparison not possible — FX conversion failed for: {', '.join(failed)}",
                    target_currency=resolved_target,
                )
            answer_note = f"FX conversion failed for: {', '.join(failed)} — shown unconverted"

        normalized: list[NormalizedFigure] = []
        for k, fig in rows:
            if fig.currency and fig.currency != resolved_target:
                rate = rate_map[(fig.currency, _normalize_date(fig.period_end))]
                amount = fig.amount * rate if rate is not None else None
                normalized.append(
                    NormalizedFigure(key=k, figure=fig, normalized_amount=amount, fx_rate=rate)
                )
            else:
                normalized.append(
                    NormalizedFigure(key=k, figure=fig, normalized_amount=fig.amount, fx_rate=None)
                )

        currency_converted = True

    else:
        normalized = [
            NormalizedFigure(key=k, figure=fig, normalized_amount=fig.amount, fx_rate=None)
            for k, fig in rows
        ]

    # Apply the comparison op. A figure with no currency or no stated scale can't be
    # compared safely, so it is left out of the ranking and named in answer_note. A key
    # with several figures (metrics or periods) has no single value to rank on.
    answer_entity: str | None = None
    keys_with_figures = {k for k, _ in rows}
    if is_comparison:
        if resolved_target is None and multi_ccy:
            raise RuntimeError(
                "argmin/argmax reached comparator with multi-currency findings "
                "and no resolved_target"
            )
        if len(rows) > len(keys_with_figures):
            answer_note = answer_note or "not ranked — several figures per entity"
        else:
            rankable = [
                n
                for n in normalized
                if n.normalized_amount is not None
                and n.figure.currency is not None
                and n.figure.unit is not None
            ]
            excluded = [n.key for n in normalized if n not in rankable]
            if excluded and answer_note is None:
                answer_note = (
                    f"excluded from ranking (unknown currency or scale): {', '.join(excluded)}"
                )
            if rankable:

                def rank_key(n: NormalizedFigure) -> float:
                    return to_millions(n.normalized_amount, n.figure.unit)  # type: ignore[arg-type]

                pick = min if op == "argmin" else max
                answer_entity = pick(rankable, key=rank_key).key

    if (
        op in ("argmin", "argmax", "list")
        and len(keys_with_figures) == 1
        and len(findings.findings) > 1
        and answer_note is None
    ):
        answer_note = "only one entity had available data"

    if chunk_texts is not None:
        # Verify the native amount — the chunk states what the filing states, never our FX
        # arithmetic. Checking the converted amount would make every converted figure
        # read as a false "not_found".
        evidence = {f.key: f.evidence for f in findings.findings if f.supported}
        normalized = [
            dc_replace(
                n,
                number_grounding=verify_value(
                    n.figure.amount,
                    n.figure.unit,
                    [chunk_texts[c] for c in evidence[n.key] if c in chunk_texts],
                ),
            )
            for n in normalized
        ]

    return ProcessedFindings(
        findings=findings,
        figures=tuple(normalized),
        answer_entity=answer_entity,
        fx_rates_used=fx_rates_used,
        currency_converted=currency_converted,
        answer_note=answer_note,
        target_currency=resolved_target,
    )


def _map_refs(raw_refs: list[str], rag_context: RAGContext) -> str:
    """Map chunk UUIDs to S-labels via the RAGContext, dropping any without a context excerpt."""
    mapped: list[str] = []
    for raw in raw_refs:
        ref: str | None = None
        with contextlib.suppress(ValueError):
            ref = rag_context.ref_for(UUID(raw))
        if ref is not None:
            mapped.append(ref)
        else:
            CITATION_REFS_DROPPED.inc()
    return ", ".join(mapped) or "—"


def _amount(currency: str | None, amount: float, unit: str | None) -> str:
    """`amount` with its currency and scale; an unstated scale is said, never assumed."""
    text = f"{currency + ' ' if currency else ''}{amount:,.1f}{unit or ''}"
    return text if unit is not None else f"{text} (scale not stated)"


def _render_figure(n: NormalizedFigure, target_currency: str | None) -> str:
    fig = n.figure
    period = " / ".join(p for p in (fig.fiscal_label, fig.period_end) if p) or "period not stated"
    native = _amount(fig.currency, fig.amount, fig.unit)
    if n.fx_rate is not None and n.normalized_amount is not None:
        approx = "" if _normalize_date(fig.period_end) else " (approx — date unavailable)"
        value = (
            f"{_amount(target_currency, n.normalized_amount, fig.unit)} | from {native}"
            f" | rate: {n.fx_rate:.4f}{approx}"
        )
    else:
        value = native
    # Only the anomaly is worth a marker — flagging every row trains the synthesis model
    # to skip it. Advisory rather than a filter: see `number_grounding.py`.
    flag = (
        " | ⚠ UNVERIFIED: value not located in cited excerpt"
        if n.number_grounding is NumberGrounding.NOT_FOUND
        else ""
    )
    return f"   - {fig.metric} ({period}): {value}{flag}"


def _render_findings_block(processed: ProcessedFindings, rag_context: RAGContext) -> str:
    findings = processed.findings
    lines = ["[FINDINGS]"]

    header_parts = []
    if processed.target_currency:
        header_parts.append(f"Target currency: {processed.target_currency}")
    if findings.comparison_op and findings.comparison_op != "none":
        header_parts.append(f"Operation: {findings.comparison_op}")
    if header_parts:
        lines.append(" | ".join(header_parts))

    if processed.answer_entity:
        best = processed.figures_for(processed.answer_entity)
        if best and best[0].normalized_amount is not None:
            cur = processed.target_currency or best[0].figure.currency
            amount = _amount(cur, best[0].normalized_amount, best[0].figure.unit)
            lines.append(f"Answer: {processed.answer_entity} ({amount})")
        else:
            lines.append(f"Answer: {processed.answer_entity}")

    if processed.fx_rates_used:
        fx_parts = [f"{k}: {v:.4f}" for k, v in processed.fx_rates_used.items()]
        lines.append("FX rates used: " + " | ".join(fx_parts))

    if processed.answer_note:
        lines.append(f"Note: {processed.answer_note}")

    lines.append("")

    for i, f in enumerate(findings.findings, 1):
        # A stated negative has no evidence to cite and no confidence worth reporting —
        # rendering it as a low-confidence claim would invite the synthesis model to
        # hedge it into a weak positive instead of reporting the absence.
        if not f.supported:
            lines.append(f"{i}. {f.key} [not disclosed] {f.claim}")
            continue
        # Drop refs with no excerpt in the synthesis context — leaking a raw ref here
        # would let the model cite an ID the citation pipeline can't resolve.
        refs = _map_refs(f.evidence, rag_context)
        lines.append(f"{i}. {f.key} [{f.confidence} confidence] {f.claim} | evidence: {refs}")
        lines.extend(
            _render_figure(n, processed.target_currency) for n in processed.figures_for(f.key)
        )

    if findings.conclusion:
        lines.append(f"\nConclusion: {findings.conclusion}")

    if findings.unresolved:
        lines.append(f"Unresolved: {'; '.join(findings.unresolved)}")

    lines.append("[END FINDINGS]")
    return "\n".join(lines)
