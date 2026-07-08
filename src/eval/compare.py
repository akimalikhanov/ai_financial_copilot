from __future__ import annotations

import json
from pathlib import Path


def _load(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def compare(prev_path: str | Path, curr_path: str | Path) -> list[str]:
    prev = _load(prev_path)
    curr = _load(curr_path)

    print(f"\n{'=' * 60}")
    print(f"COMPARE: {Path(prev_path).name}  →  {Path(curr_path).name}")
    print(f"{'=' * 60}")

    # Aggregate deltas
    prev_agg = prev.get("aggregate", {})
    curr_agg = curr.get("aggregate", {})
    _print_section_deltas("retrieval", prev_agg.get("retrieval", {}), curr_agg.get("retrieval", {}))
    _print_section_deltas(
        "correctness.by_kind",
        _flatten_by_kind(prev_agg.get("correctness", {})),
        _flatten_by_kind(curr_agg.get("correctness", {})),
    )
    _print_section_deltas("judge", prev_agg.get("judge", {}), curr_agg.get("judge", {}))
    _print_section_deltas(
        "hallucination", prev_agg.get("hallucination", {}), curr_agg.get("hallucination", {})
    )

    # Per-question regressions
    prev_by_qid = {q["qid"]: q for q in prev.get("per_question", [])}
    curr_by_qid = {q["qid"]: q for q in curr.get("per_question", [])}
    regressions: list[str] = []

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
        if pk5 - ck5:
            msgs.append(f"page_key_lost@5: {pk5 - ck5}")

        if msgs:
            regressions.append(f"  {qid}: {'; '.join(msgs)}")

    if regressions:
        print(f"\nREGRESSIONS ({len(regressions)}):")
        for r in regressions:
            print(r)
    else:
        print("\nNo regressions detected.")
    print()

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


def _print_section_deltas(name: str, prev: dict, curr: dict) -> None:
    if not prev and not curr:
        return
    keys = sorted(set(prev) | set(curr))
    rows: list[tuple[str, str]] = []
    for k in keys:
        pv = prev.get(k)
        cv = curr.get(k)
        if pv is None or cv is None:
            rows.append((k, f"{pv} → {cv}"))
        elif isinstance(pv, float) and isinstance(cv, float):
            delta = cv - pv
            sign = "+" if delta >= 0 else ""
            rows.append((k, f"{pv:.4f} → {cv:.4f}  ({sign}{delta:.4f})"))
        else:
            rows.append((k, f"{pv} → {cv}"))
    if rows:
        print(f"\n[{name}]")
        for k, v in rows:
            print(f"  {k:<35} {v}")
