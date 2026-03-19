"""
Chat worker: Celery entry point for chat queue.

Run as: .venv/bin/python -m src.workers.chat_worker
"""

from __future__ import annotations

import asyncio
import sys

from celery.signals import worker_process_init

from src.api.logging import configure_worker_logging
from src.celery_app import celery_app


@worker_process_init.connect
def _on_worker_process_init(**_kwargs: object) -> None:
    configure_worker_logging()


if __name__ == "__main__":
    configure_worker_logging()
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    celery_app.worker_main(argv=["worker", "-Q", "chat", "--pool=prefork"])
