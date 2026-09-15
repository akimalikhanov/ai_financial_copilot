"""Multiprocess metric shards: delta gauges must not survive a SIGKILLed process.

The failure this guards against was observed under load: after an ingestion worker was
OOMKilled, `celery_tasks_in_flight` read 4 with zero documents running. A killed child
never runs its `.dec()` (no `task_postrun`) and never runs `mark_process_dead` (no
`worker_process_shutdown`), so its `+1` stays in a `gauge_livesum_<pid>.db` file. The K8s
volume is an emptyDir, which is pod-scoped and therefore survives a container restart —
and because shards are named only by PID, a new child can reopen a dead one's file and
inherit the leaked value.

These tests run the real prometheus_client multiprocess machinery against a tmp dir rather
than mocking it, because the bug lives in that machinery's semantics.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from prometheus_client import CollectorRegistry, multiprocess

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_gauge(mpdir: Path, metric: str) -> dict[tuple[tuple[str, str], ...], float]:
    """Aggregate `metric`'s samples from the shard dir, exactly as a scrape would."""
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry, path=str(mpdir))
    out: dict[tuple[tuple[str, str], ...], float] = {}
    for family in registry.collect():
        if family.name != metric:
            continue
        for sample in family.samples:
            out[tuple(sorted(sample.labels.items()))] = sample.value
    return out


def _run_child(mpdir: Path, body: str) -> None:
    """Run `body` in a fresh interpreter that exits via os._exit (stands in for SIGKILL).

    A subprocess is the only honest way to produce an orphaned shard: the value has to be
    written by a process that then dies without unwinding.
    """
    script = textwrap.dedent(f"""
        import os, sys
        sys.path.insert(0, {str(REPO_ROOT)!r})
        {textwrap.indent(textwrap.dedent(body), " " * 8).lstrip()}
    """)
    env = {**os.environ, "PROMETHEUS_MULTIPROC_DIR": str(mpdir)}
    result = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, f"child failed: {result.stdout}\n{result.stderr}"


@pytest.fixture
def mpdir(tmp_path: Path) -> Path:
    d = tmp_path / "prom"
    d.mkdir()
    return d


def test_sigkilled_child_leaks_an_in_flight_increment(mpdir: Path) -> None:
    """Baseline: reproduce the leak, so the purge test below is proving something real."""
    _run_child(
        mpdir,
        """
        from prometheus_client import Gauge
        g = Gauge('celery_tasks_in_flight', 'd', ['task_name'], multiprocess_mode='livesum')
        g.labels('ingest_document').inc()
        os._exit(0)   # no .dec(), no worker_process_shutdown -- i.e. SIGKILL
        """,
    )

    samples = _read_gauge(mpdir, "celery_tasks_in_flight")
    assert samples[(("task_name", "ingest_document"),)] == 1.0, (
        "expected the orphaned increment to survive; if this fails the reproduction is stale"
    )
    assert list(mpdir.glob("gauge_livesum_*.db")), "shard file should still be on disk"


def test_purge_clears_stale_shards(mpdir: Path) -> None:
    """The fix: purging at startup makes the next process start from zero."""
    _run_child(
        mpdir,
        """
        from prometheus_client import Gauge
        g = Gauge('celery_tasks_in_flight', 'd', ['task_name'], multiprocess_mode='livesum')
        g.labels('ingest_document').inc()
        os._exit(0)
        """,
    )
    assert _read_gauge(mpdir, "celery_tasks_in_flight")  # leaked

    from src.observability import multiproc

    monkey_dir = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    try:
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = str(mpdir)
        multiproc._purged = False  # the guard is per-process; reset for the test
        multiproc.purge_multiproc_dir()
    finally:
        if monkey_dir is None:
            os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)
        else:
            os.environ["PROMETHEUS_MULTIPROC_DIR"] = monkey_dir

    assert _read_gauge(mpdir, "celery_tasks_in_flight") == {}
    assert not list(mpdir.glob("*.db"))


def test_purge_is_idempotent_and_safe_without_the_env_var() -> None:
    """Unset env var must be a no-op, not an error — `make api` and tests run that way."""
    from src.observability import multiproc

    saved = os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)
    try:
        multiproc._purged = False
        multiproc.purge_multiproc_dir()
        multiproc.purge_multiproc_dir()
    finally:
        if saved is not None:
            os.environ["PROMETHEUS_MULTIPROC_DIR"] = saved


def test_purge_runs_only_once_per_process(mpdir: Path) -> None:
    """The guard must hold: a second call would delete shards that live children hold open.

    Prefork children inherit `_purged=True` through fork, which is what stops a child from
    wiping its siblings' values.
    """
    from src.observability import multiproc

    saved = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    try:
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = str(mpdir)
        multiproc._purged = False
        multiproc.purge_multiproc_dir()

        # A "live child" writes its shard after the one legitimate purge.
        (mpdir / "gauge_livesum_4242.db").write_bytes(b"")
        multiproc.purge_multiproc_dir()

        assert (mpdir / "gauge_livesum_4242.db").exists(), (
            "second purge deleted a live child's shard"
        )
    finally:
        if saved is None:
            os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)
        else:
            os.environ["PROMETHEUS_MULTIPROC_DIR"] = saved


@pytest.mark.parametrize(
    "metric_attr,expected",
    [
        ("CELERY_TASKS_IN_FLIGHT", "livesum"),
        ("HTTP_IN_PROGRESS", "livesum"),
        ("SSE_STREAMS_OPEN", "livesum"),
        ("CELERY_QUEUE", "livemostrecent"),
    ],
)
def test_gauges_declare_a_live_multiprocess_mode(metric_attr: str, expected: str) -> None:
    """Every Gauge must opt out of the default "all" mode.

    Under "all" the pid label is retained (unbounded cardinality across restarts, and every
    query needs a manual sum()), and `mark_process_dead` cannot reach the shard at all —
    it only unlinks `gauge_live*` files.
    """
    from src.observability import metrics

    gauge = getattr(metrics, metric_attr)
    assert gauge._multiprocess_mode == expected
