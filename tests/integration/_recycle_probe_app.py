"""Celery app for test_worker_recycle: the shared app, plus one probe task and signal recorders.

Loaded by a real worker subprocess (``celery -A tests.integration._recycle_probe_app``). The
ingestion task module is imported first so its init/shutdown handlers are connected before the
recorders below, which then see the state those handlers leave behind.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from celery.signals import worker_process_init, worker_process_shutdown

from src.celery_app import celery_app
from src.services.ingestion import tasks as ingestion_tasks

PROBE_TASK = "recycle_probe"


def _record(event: str, **fields: object) -> None:
    out = Path(os.environ["RECYCLE_PROBE_DIR"]) / "events.jsonl"
    with out.open("a") as f:
        f.write(json.dumps({"event": event, "pid": os.getpid(), **fields}) + "\n")


def _live_metric_files() -> list[str]:
    mpdir = Path(os.environ["PROMETHEUS_MULTIPROC_DIR"])
    return sorted(p.name for p in mpdir.glob(f"gauge_live*_{os.getpid()}.db"))


@worker_process_init.connect
def _record_init(**_kwargs: object) -> None:
    _record("init", loop_open=ingestion_tasks._worker_loop is not None)


@worker_process_shutdown.connect
def _record_shutdown(**_kwargs: object) -> None:
    _record(
        "shutdown",
        loop_closed=ingestion_tasks._worker_loop is None,
        engine_closed=ingestion_tasks._engine is None,
        redis_closed=ingestion_tasks._redis_ingestion is None,
        live_metric_files=_live_metric_files(),
    )


@celery_app.task(name=PROBE_TASK)
def recycle_probe(tag: str) -> int:
    _record("run", tag=tag, live_metric_files=_live_metric_files())
    return os.getpid()
