from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.eval.schemas import PerQuestionResult, RunOutput
from src.models.canary_run import CanaryRun
from src.models.canary_run_result import CanaryRunResult


def run_output_to_canary_run_kwargs(output: RunOutput, run_kind: str = "agentic") -> dict:
    """Map a RunOutput → canary_runs column kwargs."""
    manifest = output.manifest
    aggregate = output.aggregate
    correctness = aggregate.correctness or {}
    hallucination = aggregate.hallucination or {}

    total_cost = sum(
        q.usage["cost_usd"] for q in output.per_question if q.usage and "cost_usd" in q.usage
    )
    total_latency = sum(q.latency_s for q in output.per_question if q.latency_s is not None)

    return {
        "run_kind": run_kind,
        "run_timestamp": datetime.fromisoformat(manifest.timestamp),
        "git_sha": manifest.git_sha,
        "test_set": manifest.test_set,
        "test_set_hash": manifest.test_set_hash,
        "model": manifest.model,
        "judge_model": manifest.judge_model,
        "k_values": manifest.k_values,
        "total_questions": manifest.total_questions,
        "evaluated": manifest.evaluated,
        "excluded": [e.model_dump() for e in manifest.excluded],
        "retrieval": aggregate.retrieval,
        "correctness": correctness,
        "correctness_overall": correctness.get("overall"),
        "judge": aggregate.judge,
        "hallucination": hallucination,
        "hallucination_rate_mean": hallucination.get("rate_mean"),
        "total_cost_usd": Decimal(str(total_cost)) if total_cost else None,
        "total_latency_s": total_latency or None,
        "raw_manifest": manifest.model_dump(mode="json"),
    }


def per_question_to_result_kwargs(pq: PerQuestionResult) -> dict:
    """Map a PerQuestionResult → canary_run_results column kwargs."""
    usage = pq.usage or {}
    judge = pq.judge or {}
    return {
        "qid": pq.qid,
        "question": pq.question,
        "kind": pq.kind,
        "route": pq.route,
        "excluded_reason": pq.excluded_reason,
        "retrieved_page_keys": pq.retrieved_page_keys,
        "metrics": pq.metrics,
        "answer": pq.answer,
        "citation_spans": pq.citation_spans,
        "correct": pq.correctness.correct if pq.correctness else None,
        "correctness_reason": pq.correctness.reason if pq.correctness else None,
        "judge": pq.judge,
        "hallucination_rate": judge.get("hallucination_rate"),
        "latency_s": pq.latency_s,
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "cost_usd": Decimal(str(usage["cost_usd"])) if usage.get("cost_usd") is not None else None,
    }


class CanaryRunRepository:
    """Repository for canary/eval run persistence."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_run(
        self,
        output: RunOutput,
        run_kind: str = "agentic",
        regressions: list[str] | None = None,
    ) -> CanaryRun:
        """Insert a canary_runs row + one canary_run_results row per question."""
        run = CanaryRun(
            **run_output_to_canary_run_kwargs(output, run_kind=run_kind),
            regressions=regressions or [],
        )
        self.session.add(run)
        await self.session.flush()

        for pq in output.per_question:
            result = CanaryRunResult(
                canary_run_id=run.id,
                **per_question_to_result_kwargs(pq),
            )
            self.session.add(result)

        await self.session.flush()
        return run

    async def list_recent(self, run_kind: str = "agentic", limit: int = 20) -> list[CanaryRun]:
        """List recent runs of a given kind, newest first."""
        result = await self.session.execute(
            select(CanaryRun)
            .where(CanaryRun.run_kind == run_kind)
            .order_by(CanaryRun.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_latest(self, run_kind: str = "agentic") -> CanaryRun | None:
        """Get the most recent run of a given kind."""
        result = await self.session.execute(
            select(CanaryRun)
            .where(CanaryRun.run_kind == run_kind)
            .order_by(CanaryRun.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_by_git_sha(self, git_sha: str, run_kind: str = "agentic") -> CanaryRun | None:
        """Get a run by git_sha and kind."""
        result = await self.session.execute(
            select(CanaryRun).where(CanaryRun.git_sha == git_sha, CanaryRun.run_kind == run_kind)
        )
        return result.scalar_one_or_none()

    async def get_with_results(self, run_id: UUID) -> CanaryRun | None:
        """Get a run with its per-question results eagerly loaded."""
        result = await self.session.execute(
            select(CanaryRun).where(CanaryRun.id == run_id).options(selectinload(CanaryRun.results))
        )
        return result.scalar_one_or_none()
