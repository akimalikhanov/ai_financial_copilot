"""
Integration test: the ingestion worker's memory recycle.

Requires: redis-broker running.

Runs a real prefork worker with --max-memory-per-child=1 KiB, so billiard's peak-RSS check
replaces the child after every task, through the same code path a large document takes.
Checks that the recycle is clean:
- each task runs in its own child, once, and is not redelivered
- the recycled child ran the ingestion shutdown handler (loop, engine, Redis closed)
- its Prometheus live-gauge files were retired
- the replacement child ran the ingestion init handler
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

import pytest
from redis import Redis

from src.celery_app import celery_app
from src.utils.config import get_redis_broker_url

REPO_ROOT = Path(__file__).resolve().parents[2]
PROBE_TASK = "recycle_probe"  # tests/integration/_recycle_probe_app.py
WORKER_TIMEOUT_S = 180


def _events(probe_dir: Path) -> list[dict]:
    path = probe_dir / "events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def _wait_for(probe_dir: Path, proc: subprocess.Popen, done) -> list[dict]:
    deadline = time.monotonic() + WORKER_TIMEOUT_S
    while time.monotonic() < deadline:
        events = _events(probe_dir)
        if done(events):
            return events
        if proc.poll() is not None:
            pytest.fail(f"worker exited early with {proc.returncode}")
        time.sleep(0.5)
    pytest.fail(f"timed out; events so far: {_events(probe_dir)}")


@pytest.mark.integration
def test_memory_recycle_replaces_the_child_cleanly(tmp_path: Path) -> None:
    queue = f"recycle-probe-{uuid4().hex[:8]}"
    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    log_path = tmp_path / "worker.log"
    env = {
        **os.environ,
        "RECYCLE_PROBE_DIR": str(probe_dir),
        "PROMETHEUS_MULTIPROC_DIR": str(tmp_path / "prom"),
        "PICTURE_ENRICHER_ENABLED": "false",
    }
    argv = [
        sys.executable, "-m", "celery", "-A", "tests.integration._recycle_probe_app",
        "worker", "--pool", "prefork", "--concurrency", "1",
        "--max-memory-per-child", "1",
        "-Q", queue, "-n", f"{queue}@%h", "--loglevel=info",
        "--without-heartbeat", "--without-gossip", "--without-mingle",
    ]  # fmt: skip

    with log_path.open("w") as log:
        proc = subprocess.Popen(argv, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    broker = Redis.from_url(get_redis_broker_url())
    try:
        for tag in ("a", "b"):
            celery_app.send_task(PROBE_TASK, args=[tag], queue=queue)

        # Two runs, two recycles, and the third child up (it replaces the second).
        events = _wait_for(
            probe_dir,
            proc,
            lambda ev: sum(e["event"] == "shutdown" for e in ev) >= 2
            and sum(e["event"] == "init" for e in ev) >= 3,
        )

        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=60)
        worker_log = log_path.read_text()

        runs = [e for e in events if e["event"] == "run"]
        assert sorted(e["tag"] for e in runs) == ["a", "b"], "each task runs exactly once"
        assert runs[0]["pid"] != runs[1]["pid"], "second task ran in a fresh child"
        assert worker_log.count("exceeding memory limit") >= 2, worker_log[-2000:]

        for run in runs:
            pid = run["pid"]
            init = next(e for e in events if e["event"] == "init" and e["pid"] == pid)
            shutdown = next(e for e in events if e["event"] == "shutdown" and e["pid"] == pid)
            assert init["loop_open"], "ingestion worker_process_init ran"
            assert run["live_metric_files"], "the probe saw its in-flight gauge file"
            assert shutdown == {
                "event": "shutdown",
                "pid": pid,
                "loop_closed": True,
                "engine_closed": True,
                "redis_closed": True,
                "live_metric_files": [],
            }

        # Acked before the recycle: nothing was restored to the queue on shutdown.
        assert broker.llen(queue) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        broker.delete(queue)
        broker.close()
