"""CLI entrypoint for the agentic eval harness.

Usage:
    python -m src.eval.run_agent \
        --test-set data/answers_small.json \
        [--output data/eval/runs/<auto>.json] \
        [--retrieval-only] [--compare <prev.json>] \
        [--limit N] [--model <model_id>] [--judge-model <judge_id>] \
        [--user-id <uuid>] [--k 5 10]

All flags are identical to src.eval.run (classic pipeline) so results can be
compared with --compare without changing mental models.
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

from redis.asyncio import Redis

from src.db.connection import get_session_factory, init_db, shutdown_db
from src.eval import loader
from src.eval.compare import compare as run_compare
from src.eval.matching import expand_pools_to_page_keys, resolve_doc_ids
from src.eval.metrics.correctness import Kind, score_correctness
from src.eval.metrics.judge import hallucination_rate, judge_one
from src.eval.metrics.retrieval import compute_retrieval_metrics, context_to_page_keys
from src.eval.pipeline_agent import run_one
from src.eval.run import _compute_aggregate, _print_summary
from src.eval.schemas import (
    CorrectnessResult,
    ExcludedEntry,
    PerQuestionResult,
    RunManifest,
    RunOutput,
)
from src.services.llm_router import get_router
from src.utils.config import (
    get_eval_judge_model,
    get_eval_output_dir,
    get_eval_user_id,
    get_redis_app_url,
)

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
    p = argparse.ArgumentParser(description="Agentic eval harness for AI Financial Copilot")
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
        "--prompt-version",
        default="v3_agent_synthesis",
        help="System prompt version for the synthesis step (e.g. v3_agent_synthesis, v3_bracket)",
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
        help="Reasoning effort for the synthesis model (e.g. low, medium, high)",
    )
    p.add_argument(
        "--persist-db",
        action="store_true",
        default=False,
        help="Persist this run to canary_runs/canary_run_results in Postgres",
    )
    return p.parse_args()


async def _run(args: argparse.Namespace) -> RunOutput:
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

    # One shared Redis client for the whole run to avoid per-question connection overhead
    redis = Redis.from_url(get_redis_app_url(), decode_responses=True)

    per_question: list[PerQuestionResult] = []
    excluded: list[ExcludedEntry] = []

    try:
        async with session_factory() as session:
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
                        redis=redis,
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

                # Log agent metadata per question
                if pr.agent_meta:
                    m = pr.agent_meta
                    logger.info(
                        "agent_meta qid=%s iterations=%d tool_calls=%d convergence=%s chunks=%d cost=$%.4f",
                        q.qid,
                        m.iterations,
                        m.tool_calls_total,
                        m.convergence_reason,
                        len(pr.rag_context.items) if pr.rag_context else 0,
                        m.cost_usd_total,
                    )

                if pr.rag_context and pr.rag_context.items:
                    retrieved_keys = context_to_page_keys(pr.rag_context)
                    result.retrieved_page_keys = retrieved_keys
                    if expanded_pools:
                        result.metrics = compute_retrieval_metrics(
                            pr.rag_context, expanded_pools, k_values=k_values
                        )
                elif pr.route != "retrieval":
                    result.excluded_reason = f"route={pr.route}"

                if pr.answer is not None:
                    result.answer = pr.answer
                    result.citation_spans = [
                        {"start": s.start, "end": s.end, "ref_ids": list(s.ref_ids)}
                        for s in pr.citation_spans
                    ]
                    result.correctness = CorrectnessResult(
                        **score_correctness(pr.answer, cast(Kind, q.kind), q.answers)
                    )

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
    finally:
        await redis.aclose()

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


async def _persist(output: RunOutput, run_kind: str, regressions: list[str] | None) -> None:
    from src.repository.canary_run_repository import CanaryRunRepository

    await init_db()
    try:
        async with get_session_factory()() as session:
            run = await CanaryRunRepository(session).create_run(
                output, run_kind=run_kind, regressions=regressions
            )
            await session.commit()
            logger.info("persisted canary_run id=%s", run.id)
    finally:
        await shutdown_db()


def main() -> None:
    args = _build_args()

    output = asyncio.run(_run(args))

    out_dir = get_eval_output_dir()
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = out_dir / f"agent_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output.model_dump_json(indent=2))
    logger.info("wrote run output to %s", out_path)

    _print_summary(output, out_path)

    regressions: list[str] = []
    if args.compare:
        compare_out = out_path.with_name(
            f"{out_path.stem}_vs_{Path(args.compare).stem}.compare.json"
        )
        regressions = run_compare(args.compare, out_path, compare_out)

    if args.persist_db:
        asyncio.run(_persist(output, run_kind="agentic", regressions=regressions))


if __name__ == "__main__":
    main()
