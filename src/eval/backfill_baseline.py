"""One-off backfill: persist the committed canary_baseline.json fixture as a DB row.

The baseline fixture (src/eval/fixtures/runs/canary_baseline.json) is generated and committed
manually, not produced by a --persist-db run of run_agent.py. Without this, Grafana trend queries
over canary_runs are missing their reference/first data point. Run once after this fixture
changes:

    .venv/bin/python -m src.eval.backfill_baseline
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.db.connection import get_session_factory, init_db, shutdown_db
from src.eval.schemas import RunOutput
from src.repository.canary_run_repository import CanaryRunRepository

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

BASELINE_PATH = Path("src/eval/fixtures/runs/canary_baseline.json")
RUN_KIND = "agentic_baseline"


async def _backfill() -> None:
    output = RunOutput.model_validate(json.loads(BASELINE_PATH.read_text()))
    git_sha = output.manifest.git_sha

    await init_db()
    try:
        async with get_session_factory()() as session:
            repo = CanaryRunRepository(session)
            existing = await repo.get_by_git_sha(git_sha, run_kind=RUN_KIND)
            if existing is not None:
                logger.info(
                    "baseline already backfilled: run_kind=%s git_sha=%s id=%s (skipping)",
                    RUN_KIND,
                    git_sha,
                    existing.id,
                )
                return

            run = await repo.create_run(output, run_kind=RUN_KIND, regressions=[])
            await session.commit()
            logger.info(
                "backfilled baseline: run_kind=%s git_sha=%s id=%s", RUN_KIND, git_sha, run.id
            )
    finally:
        await shutdown_db()


def main() -> None:
    import asyncio

    asyncio.run(_backfill())


if __name__ == "__main__":
    main()
