"""CLI entrypoint for the evaluation harness.

Usage:
    python -m src.eval.run \
        --test-set data/answers_small.json \
        [--output data/eval/runs/<auto>.json] \
        [--retrieval-only] [--compare <prev.json>] \
        [--limit N] [--model <model_id>] [--judge-model <judge_id>] \
        [--user-id <uuid>] [--k 5 10]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import UUID

from src.db.connection import get_session_factory, init_db, shutdown_db
from src.eval import loader
from src.eval.compare import compare as run_compare
from src.eval.matching import expand_pools_to_page_keys, resolve_doc_ids
from src.eval.metrics.correctness import Kind, score_correctness
from src.eval.metrics.judge import hallucination_rate, judge_one
from src.eval.metrics.retrieval import compute_retrieval_metrics, context_to_page_keys
from src.eval.pipeline import run_one
from src.eval.schemas import (
    AggregateMetrics,
    CorrectnessResult,
    ExcludedEntry,
    PerQuestionResult,
    RunManifest,
    RunOutput,
)
from src.services.llm_router import get_router
from src.utils.config import get_eval_judge_model, get_eval_output_dir, get_eval_user_id

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Eval harness for AI Financial Copilot")
    p.add_argument("--test-set", default="data/answers_small.json")
    p.add_argument("--output", default=None)
    p.add_argument("--retrieval-only", action="store_true")
    p.add_argument("--compare", default=None, metavar="PREV_RUN")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--qid",
        nargs="+",
        default=None,
        metavar="QID",
        help="Run only these question IDs (e.g. q025 q003)",
    )
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument(
        "--prompt-version", default="v3_bracket", help="System prompt version (e.g. v3_bracket, v1)"
    )
    p.add_argument("--judge-model", default=None)
    p.add_argument("--user-id", default=None)
    p.add_argument("--k", type=int, nargs="+", default=[5, 10])
    p.add_argument(
        "--page-tolerance",
        type=int,
        default=2,
        help="±N pages around golden page number (0 = exact match)",
    )
    p.add_argument(
        "--reasoning-effort",
        default=None,
        help="Reasoning effort for the model (e.g. low, medium, high)",
    )
    return p.parse_args()


async def _run(args: argparse.Namespace) -> RunOutput:
    # Resolve user_id
    user_id_str = args.user_id or (str(get_eval_user_id()) if get_eval_user_id() else None)
    if not user_id_str:
        raise RuntimeError("--user-id or EVAL_USER_ID is required")
    user_id = UUID(user_id_str)

    judge_model = args.judge_model or get_eval_judge_model()
    k_values = tuple(args.k)
    test_set_path = Path(args.test_set)

    questions = loader.load(test_set_path)
    if args.qid:
        questions = [q for q in questions if q.qid in args.qid]
        if not questions:
            raise RuntimeError(f"No questions matched --qid {args.qid}")
    elif args.limit:
        questions = questions[: args.limit]

    await init_db()
    session_factory = get_session_factory()
    llm_router = get_router()

    per_question: list[PerQuestionResult] = []
    excluded: list[ExcludedEntry] = []

    async with session_factory() as session:
        # Resolve all doc_ids upfront from all pools
        all_pools = [pool for q in questions for pool in q.reference_pools]
        resolver = await resolve_doc_ids(all_pools, session, user_id)

        for q in questions:
            t0 = time.monotonic()
            result = PerQuestionResult(
                qid=q.qid,
                question=q.question,
                kind=q.kind,
                expected_answers=q.answers,
                reference_pools=q.reference_pools,
            )

            # Check for unresolved titles
            unresolved = [
                entry
                for pool in q.reference_pools
                for entry in pool
                if resolver.get(entry) is None and q.reference_pools
            ]
            if unresolved:
                reason = f"unresolved_title: {unresolved[0]}"
                result.excluded_reason = reason
                excluded.append(ExcludedEntry(qid=q.qid, reason=reason))
                logger.warning("excluded qid=%s reason=%s", q.qid, reason)
                per_question.append(result)
                continue

            expanded_pools = expand_pools_to_page_keys(
                q.reference_pools, resolver, page_tolerance=args.page_tolerance
            )

            try:
                pr = await run_one(
                    q,
                    session=session,
                    user_id=user_id,
                    model_id=args.model,
                    prompt_version=args.prompt_version,
                    reasoning_effort=args.reasoning_effort,
                    llm_router=llm_router,
                    retrieval_only=args.retrieval_only,
                )
            except Exception:
                logger.exception("pipeline_error qid=%s", q.qid)
                reason = "pipeline_error"
                result.excluded_reason = reason
                excluded.append(ExcludedEntry(qid=q.qid, reason=reason))
                per_question.append(result)
                continue

            result.route = pr.route
            result.latency_s = round(time.monotonic() - t0, 3)

            if pr.usage:
                result.usage = {
                    "input_tokens": pr.usage.input_tokens,
                    "output_tokens": pr.usage.output_tokens,
                    "cost_usd": pr.usage.cost_usd,
                }

            # Retrieval metrics
            if pr.rag_context and pr.rag_context.items:
                retrieved_keys = context_to_page_keys(pr.rag_context)
                result.retrieved_page_keys = retrieved_keys
                if expanded_pools:
                    result.metrics = compute_retrieval_metrics(
                        pr.rag_context, expanded_pools, k_values=k_values
                    )
            elif pr.route != "retrieval":
                result.excluded_reason = f"route={pr.route}"

            # Correctness
            if pr.answer is not None:
                result.answer = pr.answer
                result.citation_spans = [
                    {"start": s.start, "end": s.end, "ref_ids": list(s.ref_ids)}
                    for s in pr.citation_spans
                ]
                result.correctness = CorrectnessResult(
                    **score_correctness(pr.answer, cast(Kind, q.kind), q.answers)
                )

            # Judge (skip when no retrieval context)
            if (
                not args.retrieval_only
                and pr.rag_context
                and pr.rag_context.items
                and pr.answer is not None
            ):
                gold = q.answers[0] if q.answers else "N/A"
                judge_out = await judge_one(
                    question=q.question,
                    rag_context=pr.rag_context,
                    answer=pr.answer,
                    gold_answer=gold,
                    model_id=judge_model,
                )
                if judge_out:
                    hal_rate = hallucination_rate(judge_out, pr.citation_spans)
                    result.judge = {
                        **judge_out.model_dump(),
                        "hallucination_rate": hal_rate,
                    }

            per_question.append(result)
            logger.info(
                "done qid=%s route=%s latency=%.1fs correct=%s",
                q.qid,
                result.route,
                result.latency_s or 0,
                result.correctness.correct if result.correctness else None,
            )

    await shutdown_db()

    aggregate = _compute_aggregate(per_question, k_values)
    manifest = RunManifest(
        timestamp=datetime.now(UTC).isoformat(),
        git_sha=_git_sha(),
        test_set=str(test_set_path),
        test_set_hash=_file_sha256(test_set_path),
        model=args.model,
        judge_model=judge_model,
        k_values=list(k_values),
        total_questions=len(questions),
        evaluated=len(questions) - len(excluded),
        excluded=excluded,
    )
    return RunOutput(manifest=manifest, aggregate=aggregate, per_question=per_question)


def _compute_aggregate(
    results: list[PerQuestionResult], _k_values: tuple[int, ...]
) -> AggregateMetrics:
    retrieval_rows = [r for r in results if r.metrics and r.excluded_reason is None]
    judge_rows = [r for r in results if r.judge and r.excluded_reason is None]
    correctness_rows = [r for r in results if r.correctness and r.excluded_reason is None]

    # Retrieval
    ret: dict[str, float] = {}
    if retrieval_rows:
        keys = list(retrieval_rows[0].metrics.keys())
        for k in keys:
            vals = [r.metrics[k] for r in retrieval_rows if k in r.metrics]
            ret[k] = round(sum(vals) / len(vals), 4) if vals else 0.0

    # Correctness
    correct_agg: dict = {}
    if correctness_rows:
        overall = sum(1 for r in correctness_rows if r.correctness and r.correctness.correct)
        correct_agg["overall"] = round(overall / len(correctness_rows), 4)
        by_kind: dict[str, dict] = {}
        for r in correctness_rows:
            kind = r.kind
            if kind not in by_kind:
                by_kind[kind] = {"n": 0, "correct": 0}
            by_kind[kind]["n"] += 1
            if r.correctness and r.correctness.correct:
                by_kind[kind]["correct"] += 1
        correct_agg["by_kind"] = {
            k: {"n": v["n"], "acc": round(v["correct"] / v["n"], 4)} for k, v in by_kind.items()
        }

    # Judge
    judge_agg: dict[str, float] = {}
    if judge_rows:
        for dim in ("faithfulness", "relevance", "citation_accuracy", "completeness"):
            scores = [
                r.judge[dim]["score"]
                for r in judge_rows
                if r.judge and dim in r.judge and isinstance(r.judge[dim], dict)
            ]
            if scores:
                judge_agg[f"{dim}_mean"] = round(sum(scores) / len(scores), 4)

    # Hallucination
    hal_agg: dict = {}
    if judge_rows:
        rates = [
            r.judge["hallucination_rate"]
            for r in judge_rows
            if r.judge and "hallucination_rate" in r.judge
        ]
        if rates:
            hal_agg["rate_mean"] = round(sum(rates) / len(rates), 4)
        hal_agg["questions_with_any_unsupported"] = sum(
            1 for r in judge_rows if r.judge and r.judge.get("unsupported_claims")
        )

    return AggregateMetrics(
        retrieval=ret,
        correctness=correct_agg,
        judge=judge_agg,
        hallucination=hal_agg,
    )


def _print_summary(output: RunOutput, out_path: Path) -> None:
    pqs = output.per_question
    agg = output.aggregate
    m = output.manifest

    latencies = [q.latency_s for q in pqs if q.latency_s is not None]
    total_cost = sum(q.usage["cost_usd"] for q in pqs if q.usage and "cost_usd" in q.usage)
    total_input = sum(q.usage["input_tokens"] for q in pqs if q.usage and "input_tokens" in q.usage)
    total_output = sum(
        q.usage["output_tokens"] for q in pqs if q.usage and "output_tokens" in q.usage
    )

    w = 54
    print(f"\n{'━' * w}")
    print(
        f"  EVAL COMPLETE — {m.evaluated}/{m.total_questions} questions  ({len(m.excluded)} excluded)"
    )
    print(f"{'━' * w}")

    if agg.retrieval:
        print("  RETRIEVAL")
        print(
            f"    precision@5 / @10   {agg.retrieval.get('precision@5', 0):.3f} / {agg.retrieval.get('precision@10', 0):.3f}"
        )
        print(
            f"    recall@5    / @10   {agg.retrieval.get('recall@5', 0):.3f} / {agg.retrieval.get('recall@10', 0):.3f}"
        )
        print(f"    MRR                 {agg.retrieval.get('mrr', 0):.3f}")
        print(f"    NDCG@10             {agg.retrieval.get('ndcg@10', 0):.3f}")

    if agg.correctness:
        print(f"  CORRECTNESS         {agg.correctness.get('overall', 0):.1%} overall")
        for kind, stats in (agg.correctness.get("by_kind") or {}).items():
            print(f"    {kind:<10}  n={stats['n']}   acc={stats['acc']:.1%}")

    if agg.judge:
        print("  JUDGE (mean 1–5)")
        for dim in ("faithfulness", "relevance", "citation_accuracy", "completeness"):
            val = agg.judge.get(f"{dim}_mean")
            if val is not None:
                print(f"    {dim:<22} {val:.2f}")

    if agg.hallucination:
        print("  HALLUCINATION")
        print(f"    rate (mean)         {agg.hallucination.get('rate_mean', 0):.3f}")
        print(
            f"    questions w/ any    {agg.hallucination.get('questions_with_any_unsupported', 0)}"
        )

    print("  COST & LATENCY")
    if latencies:
        print(f"    avg latency         {sum(latencies) / len(latencies):.1f}s")
        print(f"    total wall time     {sum(latencies):.0f}s")
    if total_cost:
        print(f"    tokens in / out     {total_input:,} / {total_output:,}")
        print(f"    total cost          ${total_cost:.4f}")

    print(f"{'━' * w}")
    print(f"  output → {out_path}")
    print(f"{'━' * w}\n")


def main() -> None:
    args = _build_args()

    if args.compare:
        # Pure compare mode — no run needed if output is also provided, but the
        # plan says --compare is an add-on flag that loads both JSONs. If the user
        # wants to compare two existing files they can pass --compare alone. We
        # detect this by checking if --output or --test-set are the defaults AND
        # --compare references a second file alongside an existing output path.
        # Simplest interpretation per spec: run the current eval AND compare to prev.
        pass  # handled after run below

    output = asyncio.run(_run(args))

    # Determine output path
    out_dir = get_eval_output_dir()
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = out_dir / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output.model_dump_json(indent=2))
    logger.info("wrote run output to %s", out_path)

    _print_summary(output, out_path)

    if args.compare:
        compare_out = out_path.with_name(
            f"{out_path.stem}_vs_{Path(args.compare).stem}.compare.json"
        )
        run_compare(args.compare, out_path, compare_out)


if __name__ == "__main__":
    main()
