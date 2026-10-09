"""
Celery worker for document ingestion. Sets document status to ready.

Run as: .venv/bin/python -m src.workers.ingestion_worker
"""

from __future__ import annotations

import logging
import os
import socket

from celery.signals import worker_process_init

from src.api.logging import configure_worker_logging
from src.celery_app import celery_app
from src.observability.metrics import INGESTION_WORKER_RECYCLE_THRESHOLD
from src.observability.worker_metrics import start_worker_metrics
from src.utils.config import get_ingest_worker_max_memory_per_child_kb

logger = logging.getLogger(__name__)


@worker_process_init.connect
def _on_worker_process_init(**_kwargs: object) -> None:
    configure_worker_logging()


def build_worker_argv(nodename: str, pool: str, concurrency: str | None) -> list[str]:
    argv = ["worker", "--loglevel=info", "--pool", pool, "-n", nodename, "-Q", "ingestion"]
    if concurrency:
        argv.extend(["--concurrency", concurrency])
    # What a child keeps after a document grows with the largest document it has parsed, so
    # the recycle is on memory, not task count. Set here and not in celery_app.conf, which the
    # chat worker shares.
    max_memory_kb = get_ingest_worker_max_memory_per_child_kb()
    if max_memory_kb > 0:
        argv.extend(["--max-memory-per-child", str(max_memory_kb)])
    return argv


if __name__ == "__main__":
    os.environ.setdefault("INGESTION_LOG_ONLY_PIPELINE", "1")
    configure_worker_logging()
    nodename = f"ingestion@{socket.gethostname()}.{os.getpid()}"
    pool = os.getenv("CELERY_WORKER_POOL", "prefork")
    concurrency = os.getenv("CELERY_WORKER_CONCURRENCY")

    argv = build_worker_argv(nodename, pool, concurrency)

    start_worker_metrics(port=9101, queues=("ingestion",))
    INGESTION_WORKER_RECYCLE_THRESHOLD.set(get_ingest_worker_max_memory_per_child_kb() * 1024)

    logger.info(
        "ingestion_worker.starting",
        extra={
            "node_name": nodename,
            "pool": pool,
            "concurrency": concurrency,
            "max_memory_per_child_kb": get_ingest_worker_max_memory_per_child_kb(),
        },
    )
    celery_app.worker_main(argv=argv)
