from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _load(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def compare(
    prev_path: str | Path, curr_path: str | Path, output_path: str | Path | None = None
) -> list[str]:
    prev = _load(prev_path)
    curr = _load(curr_path)

    print(f"\n{'=' * 60}")
    print(f"COMPARE: {Path(prev_path).name}  →  {Path(curr_path).name}")
    print(f"{'=' * 60}")

    # Aggregate deltas
    prev_agg = prev.get("aggregate", {})
    curr_agg = curr.get("aggregate", {})
    sections = {
        "retrieval": _section_deltas(prev_agg.get("retrieval", {}), curr_agg.get("retrieval", {})),
        "correctness.by_kind": _section_deltas(
            _flatten_by_kind(prev_agg.get("correctness", {})),
            _flatten_by_kind(curr_agg.get("correctness", {})),
        ),
        "judge": _section_deltas(prev_agg.get("judge", {}), curr_agg.get("judge", {})),
        "hallucination": _section_deltas(
            prev_agg.get("hallucination", {}), curr_agg.get("hallucination", {})
        ),
    }
    for name, rows in sections.items():
        _print_section(name, rows)

    # Per-question regressions
    prev_by_qid = {q["qid"]: q for q in prev.get("per_question", [])}
    curr_by_qid = {q["qid"]: q for q in curr.get("per_question", [])}
    regressions: list[str] = []
    regression_details: list[dict[str, Any]] = []

    for qid, cq in curr_by_qid.items():
        pq = prev_by_qid.get(qid)
        if pq is None:
            continue
        msgs: list[str] = []

        # Correctness flip: was correct, now wrong
        pc = (pq.get("correctness") or {}).get("correct")
        cc = (cq.get("correctness") or {}).get("correct")
        if pc is True and cc is False:
            msgs.append("correctness_flip: correct→wrong")

        # Judge score drop > 1 on any dimension
        pj = pq.get("judge") or {}
        cj = cq.get("judge") or {}
        for dim in ("faithfulness", "relevance", "citation_accuracy", "completeness"):
            ps = (pj.get(dim) or {}).get("score")
            cs = (cj.get(dim) or {}).get("score")
            if ps is not None and cs is not None and (ps - cs) > 1:
                msgs.append(f"judge_{dim}_drop: {ps}→{cs}")

        # New unsupported spans
        p_unsup = len(pj.get("unsupported_claims") or [])
        c_unsup = len(cj.get("unsupported_claims") or [])
        if c_unsup > p_unsup:
            msgs.append(f"new_unsupported_claims: {p_unsup}→{c_unsup}")

        # page_key hit lost at K=5
        pk5 = set(pq.get("retrieved_page_keys", [])[:5])
        ck5 = set(cq.get("retrieved_page_keys", [])[:5])
        lost_keys = pk5 - ck5
        if lost_keys:
            msgs.append(f"page_key_lost@5: {lost_keys}")

        if msgs:
            regressions.append(f"  {qid}: {'; '.join(msgs)}")
            regression_details.append({"qid": qid, "question": cq.get("question"), "issues": msgs})

    if regressions:
        print(f"\nREGRESSIONS ({len(regressions)}):")
        for r in regressions:
            print(r)
    else:
        print("\nNo regressions detected.")
    print()

    if output_path is not None:
        summary = {
            "compared_at": datetime.now(UTC).isoformat(),
            "prev_run": str(prev_path),
            "curr_run": str(curr_path),
            "prev_manifest": prev.get("manifest", {}),
            "curr_manifest": curr.get("manifest", {}),
            "sections": sections,
            "regressions": regression_details,
        }
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))
        print(f"comparison written to {out}\n")

    return regressions


def _flatten_by_kind(correctness: dict) -> dict:
    flat: dict = {}
    overall = correctness.get("overall")
    if overall is not None:
        flat["overall"] = overall
    for kind, stats in correctness.get("by_kind", {}).items():
        if isinstance(stats, dict) and "acc" in stats:
            flat[f"{kind}.acc"] = stats["acc"]
    return flat


def _section_deltas(prev: dict, curr: dict) -> list[dict[str, Any]]:
    keys = sorted(set(prev) | set(curr))
    rows: list[dict[str, Any]] = []
    for k in keys:
        pv = prev.get(k)
        cv = curr.get(k)
        delta = cv - pv if isinstance(pv, float) and isinstance(cv, float) else None
        rows.append({"metric": k, "prev": pv, "curr": cv, "delta": delta})
    return rows


def _print_section(name: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    print(f"\n[{name}]")
    for row in rows:
        k, pv, cv, delta = row["metric"], row["prev"], row["curr"], row["delta"]
        if delta is not None:
            sign = "+" if delta >= 0 else ""
            print(f"  {k:<35} {pv:.4f} → {cv:.4f}  ({sign}{delta:.4f})")
        else:
            print(f"  {k:<35} {pv} → {cv}")
