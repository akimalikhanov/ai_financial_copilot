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
from wsgiref.simple_server import WSGIRequestHandler, make_server

from celery.signals import task_postrun, task_prerun, task_retry, worker_process_shutdown
from prometheus_client import REGISTRY, CollectorRegistry, make_wsgi_app, multiprocess
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.registry import Collector

from src.observability.metrics import (
    CELERY_DURATION,
    CELERY_QUEUE,
    CELERY_TASKS,
    CELERY_TASKS_IN_FLIGHT,
)
from src.observability.multiproc import purge_multiproc_dir
from src.observability.process_memory import (
    cgroup_memory,
    child_pids,
    descendant_pids,
    pid_pss_bytes,
)

logger = logging.getLogger(__name__)


class _QuietWSGIRequestHandler(WSGIRequestHandler):
    """Drops the per-request access log line.

    Prometheus scrapes this server every 15s; wsgiref writes straight to stderr
    (bypassing the `logging` module), which otherwise floods pod logs with
    "GET /metrics 200" on every scrape. Errors still surface via handle_error.
    """

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 — stdlib signature
        pass


# task_id -> start perf_counter, to measure duration in task_postrun
_task_starts: dict[str, float] = {}

_QUEUE_SAMPLE_INTERVAL = float(os.getenv("CELERY_QUEUE_SAMPLE_INTERVAL_SECONDS", "15"))

_started = False


@task_prerun.connect
def _on_task_prerun(task_id: str | None = None, task=None, **_kwargs: object) -> None:
    if task_id is not None:
        _task_starts[task_id] = time.perf_counter()
    CELERY_TASKS_IN_FLIGHT.labels(getattr(task, "name", "unknown")).inc()


@task_postrun.connect
def _on_task_postrun(
    task_id: str | None = None, task=None, state: str | None = None, **_kwargs: object
) -> None:
    name = getattr(task, "name", "unknown")
    CELERY_TASKS_IN_FLIGHT.labels(name).dec()
    CELERY_TASKS.labels(name, (state or "UNKNOWN").lower()).inc()
    start = _task_starts.pop(task_id, None) if task_id is not None else None
    if start is not None:
        CELERY_DURATION.labels(name).observe(time.perf_counter() - start)


@task_retry.connect
def _on_task_retry(sender=None, **_kwargs: object) -> None:
    CELERY_TASKS.labels(getattr(sender, "name", "unknown"), "retry").inc()


@worker_process_shutdown.connect
def _on_worker_process_shutdown_metrics(**_kwargs: object) -> None:
    """Retire this child's multiprocess metric files.

    ``livesum`` sums every live PID's file and drops dead ones — but only once
    they are marked dead; it does not detect exits by itself. Without this, a
    child that exits mid-task leaves its ``celery_tasks_in_flight`` increment in
    place forever, and the gauge ratchets upward across pool recycles. Only
    covers a graceful exit; a SIGKILLed child still leaks until the pod restarts.
    """
    mpdir = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    if not mpdir:
        return
    try:
        multiprocess.mark_process_dead(os.getpid(), mpdir)
    except Exception:  # noqa: BLE001 — never let metrics cleanup block shutdown
        logger.debug("worker_metrics.mark_process_dead_failed", exc_info=True)


class ProcessTreeMemoryCollector(Collector):
    """Scrape-time memory of the worker's process tree and its container.

    Runs in the parent, which serves /metrics, so the pool children need no sampler thread.
    Uses PSS, so the roles add up and the cgroup's ``current`` minus their sum is memory no
    process owns: page cache and kernel memory.

    role="descendants" is what the pool children start themselves, such as torch.compile's
    compile-worker pool: the cgroup pays for it, and nothing else reports it.
    """

    def collect(self):
        pss = GaugeMetricFamily(
            "worker_process_pss_bytes",
            "PSS of the worker parent, its pool children, and their descendants",
            labels=["role"],
        )
        pss.add_metric(["parent"], float(pid_pss_bytes(os.getpid()) or 0))
        # A zombie (a reaped-late child) has no memory map, which also keeps it out of the count.
        children = {pid: m for pid in child_pids() if (m := pid_pss_bytes(pid)) is not None}
        pss.add_metric(["children"], float(sum(children.values())))
        descendants = (pid_pss_bytes(d) for pid in children for d in descendant_pids(pid))
        pss.add_metric(["descendants"], float(sum(m for m in descendants if m is not None)))
        yield pss
        yield GaugeMetricFamily(
            "worker_pool_children", "Live pool children", value=float(len(children))
        )
        cgroup = cgroup_memory()
        if cgroup:
            family = GaugeMetricFamily(
                "worker_cgroup_memory_bytes",
                "Container cgroup v2 memory: current, and memory.stat anon/file/inactive_file/shmem",
                labels=["kind"],
            )
            for kind, value in cgroup.items():
                family.add_metric([kind], float(value))
            yield family


def _sample_queue_depth(queues: tuple[str, ...]) -> None:
    """Periodically sample broker list length per queue into CELERY_QUEUE.

    LLEN counts *waiting* tasks only. A task that a worker has reserved is gone
    from the list, so in-flight and unacked work is invisible here: with one
    ingestion slot, an idle worker and a worker mid-parse both read 0. Pair this
    with ``celery_tasks_in_flight`` — total outstanding work is the sum of the
    two, and "is anything running" is the second one alone.

    Uses a sync Redis client on a daemon thread to stay off the worker's event loop.
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

    Purges stale shards first: this runs in the parent before any child is forked,
    which is the only safe moment to delete them. Normally already done by the
    ``src.observability`` package body; the call is idempotent and kept here because
    the aggregation below is only correct on a clean directory.
    """
    global _started
    if _started:
        return
    _started = True

    purge_multiproc_dir()

    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
    else:
        registry = REGISTRY
    registry.register(ProcessTreeMemoryCollector())
    app = make_wsgi_app(registry)

    httpd = make_server("", port, app, handler_class=_QuietWSGIRequestHandler)
    threading.Thread(target=httpd.serve_forever, name="metrics-server", daemon=True).start()

    threading.Thread(
        target=_sample_queue_depth, args=(queues,), name="celery-queue-sampler", daemon=True
    ).start()
    logger.info("worker_metrics.started", extra={"port": port, "queues": list(queues)})
