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
from src.services.ingestion.celery_app import celery_app

logger = logging.getLogger(__name__)


@worker_process_init.connect
def _on_worker_process_init(**_kwargs: object) -> None:
    configure_worker_logging()


if __name__ == "__main__":
    os.environ.setdefault("INGESTION_LOG_ONLY_PIPELINE", "1")
    configure_worker_logging()
    nodename = f"ingestion@{socket.gethostname()}.{os.getpid()}"
    pool = os.getenv("CELERY_WORKER_POOL", "prefork")
    concurrency = os.getenv("CELERY_WORKER_CONCURRENCY")

    argv = ["worker", "--loglevel=info", "--pool", pool, "-n", nodename]
    if concurrency:
        argv.extend(["--concurrency", concurrency])

    logger.info(
        "ingestion_worker.starting",
        extra={
            "node_name": nodename,
            "pool": pool,
            "concurrency": concurrency,
        },
    )
    celery_app.worker_main(argv=argv)
