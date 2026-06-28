"""Prometheus metrics exposition for Celery workers.

Workers aren't HTTP-scraped like the API, so each worker process starts its own
``/metrics`` HTTP server and updates task counters via Celery signals. A small
background thread samples broker queue depth.

Call :func:`start_worker_metrics` once per worker process (from the bootstrap
``worker_process_init`` hook), passing the port and the queues to sample.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from wsgiref.simple_server import make_server

from celery.signals import task_postrun, task_prerun, task_retry
from prometheus_client import CollectorRegistry, make_wsgi_app, multiprocess, start_http_server

from src.observability.metrics import CELERY_DURATION, CELERY_QUEUE, CELERY_TASKS

logger = logging.getLogger(__name__)

# task_id -> start perf_counter, to measure duration in task_postrun
_task_starts: dict[str, float] = {}

_QUEUE_SAMPLE_INTERVAL = float(os.getenv("CELERY_QUEUE_SAMPLE_INTERVAL_SECONDS", "15"))

_started = False


@task_prerun.connect
def _on_task_prerun(task_id: str | None = None, **_kwargs: object) -> None:
    if task_id is not None:
        _task_starts[task_id] = time.perf_counter()


@task_postrun.connect
def _on_task_postrun(
    task_id: str | None = None, task=None, state: str | None = None, **_kwargs: object
) -> None:
    name = getattr(task, "name", "unknown")
    CELERY_TASKS.labels(name, (state or "UNKNOWN").lower()).inc()
    start = _task_starts.pop(task_id, None) if task_id is not None else None
    if start is not None:
        CELERY_DURATION.labels(name).observe(time.perf_counter() - start)


@task_retry.connect
def _on_task_retry(sender=None, **_kwargs: object) -> None:
    CELERY_TASKS.labels(getattr(sender, "name", "unknown"), "retry").inc()


def _sample_queue_depth(queues: tuple[str, ...]) -> None:
    """Periodically sample broker list length per queue into CELERY_QUEUE.

    LLEN is approximate (omits in-flight/unacked tasks) — good enough for trend
    and alerting. Uses a sync Redis client on a daemon thread to stay off the
    worker's event loop.
    """
    from redis import Redis

    from src.utils.config import get_redis_broker_url

    client = Redis.from_url(get_redis_broker_url())
    while True:
        for queue in queues:
            try:
                CELERY_QUEUE.labels(queue).set(int(client.llen(queue)))  # type: ignore[arg-type]
            except Exception:  # noqa: BLE001 — never let sampling crash the worker
                logger.debug("celery_queue_sample_failed", extra={"queue": queue})
        time.sleep(_QUEUE_SAMPLE_INTERVAL)


def start_worker_metrics(port: int, queues: tuple[str, ...]) -> None:
    """Start the metrics HTTP server and queue-depth sampler.

    Call once from the worker's parent process (``__main__``), before forking the
    prefork pool. Prefork children share metrics via ``PROMETHEUS_MULTIPROC_DIR``
    (set in the worker bootstrap); the parent's server aggregates them with a
    multiprocess collector. Without that env var (e.g. solo pool) it serves the
    default single-process registry.
    """
    global _started
    if _started:
        return
    _started = True

    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        app = make_wsgi_app(registry)
        httpd = make_server("", port, app)
        threading.Thread(target=httpd.serve_forever, name="metrics-server", daemon=True).start()
    else:
        start_http_server(port)

    threading.Thread(
        target=_sample_queue_depth, args=(queues,), name="celery-queue-sampler", daemon=True
    ).start()
    logger.info("worker_metrics.started", extra={"port": port, "queues": list(queues)})
