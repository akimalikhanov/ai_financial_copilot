"""Ingestion worker argv: the memory recycle that bounds what a worker process carries."""

from __future__ import annotations

import pytest

from src.utils.config import get_ingest_max_pages, get_ingest_worker_max_memory_per_child_kb
from src.workers.ingestion_worker import build_worker_argv

GIB = 1024**3


def _option(argv: list[str], name: str) -> str | None:
    return argv[argv.index(name) + 1] if name in argv else None


class TestBuildWorkerArgv:
    def test_passes_the_memory_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("INGEST_WORKER_MAX_MEMORY_PER_CHILD_KB", "2048")
        argv = build_worker_argv("n", "prefork", "1")
        assert _option(argv, "--max-memory-per-child") == "2048"
        assert _option(argv, "--concurrency") == "1"

    def test_defaults_to_5_gb(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("INGEST_WORKER_MAX_MEMORY_PER_CHILD_KB", raising=False)
        argv = build_worker_argv("n", "prefork", None)
        assert _option(argv, "--max-memory-per-child") == "4882812"

    def test_zero_disables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("INGEST_WORKER_MAX_MEMORY_PER_CHILD_KB", "0")
        assert "--max-memory-per-child" not in build_worker_argv("n", "prefork", None)

    def test_never_recycles_on_task_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # D1: a task count does not bound the floor, retention follows the peak.
        monkeypatch.delenv("INGEST_WORKER_MAX_MEMORY_PER_CHILD_KB", raising=False)
        assert "--max-tasks-per-child" not in build_worker_argv("n", "prefork", "1")

    def test_shared_app_config_has_no_recycle(self) -> None:
        # The chat worker shares celery_app; the recycle belongs on the ingestion argv only.
        from src.celery_app import celery_app

        assert not celery_app.conf.worker_max_memory_per_child
        assert not celery_app.conf.worker_max_tasks_per_child


def test_default_budget_fits_the_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """T + parent + demand(P_max) + margin <= limit, with the shipped defaults.

    Fails when the page gate or T is raised without the other.

    demand is per-page and measured on a warm child that returns its memory after each
    document: 1043 pages needed 5.13 GB on top of the floor it started from. The earlier
    7.55 GB at the same page count was a child that kept everything it allocated.
    """
    monkeypatch.delenv("INGEST_MAX_PAGES", raising=False)
    monkeypatch.delenv("INGEST_WORKER_MAX_MEMORY_PER_CHILD_KB", raising=False)
    limit_gb = 12 * GIB / 1e9
    margin_gb = 1.0
    parent_gb = 0.48
    demand_gb = 5.13 / 1043 * get_ingest_max_pages()
    threshold_gb = get_ingest_worker_max_memory_per_child_kb() * 1024 / 1e9

    assert threshold_gb + parent_gb + demand_gb + margin_gb <= limit_gb
