"""
Celery worker for document ingestion. Sets document status to ready.

Run as: .venv/bin/python -m src.workers.ingestion_worker
"""

from __future__ import annotations

import os
import socket

from celery.signals import worker_process_init

from src.api.logging import configure_worker_logging
from src.services.ingestion.celery_app import celery_app


@worker_process_init.connect
def _on_worker_process_init(**_kwargs: object) -> None:
    configure_worker_logging()


if __name__ == "__main__":
    configure_worker_logging()
    nodename = f"ingestion@{socket.gethostname()}.{os.getpid()}"
    celery_app.worker_main(argv=["worker", "--loglevel=info", "-n", nodename])
