"""ingest_document's memory instrumentation."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, cast

import pytest
from prometheus_client import REGISTRY

from src.observability.process_memory import TaskMemory
from src.services.ingestion import tasks


@pytest.fixture(autouse=True)
def worker_loop(monkeypatch: pytest.MonkeyPatch):
    loop = asyncio.new_event_loop()
    monkeypatch.setattr(tasks, "_worker_loop", loop)
    monkeypatch.setattr(tasks, "_child_tasks", 0)
    yield loop
    loop.close()


def _memory_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.getMessage() == "pipeline.memory"]


def _ingest(document_id: str) -> None:
    cast(Any, tasks.ingest_document).run(document_id)


def _gauge(name: str, **labels: str) -> float | None:
    return REGISTRY.get_sample_value(name, labels)


def test_logs_floors_and_stage_peaks(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def fake_pipeline(_document_id: str, mem: TaskMemory | None = None) -> None:
        assert mem is not None
        mem.stage("parse_pdf_docling")
        mem.page_count = 7
        mem.stage("finalize_ready")

    monkeypatch.setattr(tasks, "_run_pipeline", fake_pipeline)
    monkeypatch.setenv("INGEST_WORKER_MAX_MEMORY_PER_CHILD_KB", "1")

    recycles_before = _gauge("ingestion_worker_recycles_total") or 0
    with caplog.at_level(logging.INFO, logger=tasks.logger.name):
        _ingest("doc-1")
        _ingest("doc-2")

    records = _memory_records(caplog)
    assert [r.document_id for r in records] == ["doc-1", "doc-2"]  # type: ignore[attr-defined]
    last = records[-1].__dict__
    assert last["child_tasks"] == 2
    assert last["page_count"] == 7
    assert set(last["stage_peaks_mb"]) == {"parse_pdf_docling", "finalize_ready"}
    assert set(last["stage_growth_mb"]) == {"parse_pdf_docling", "finalize_ready"}
    assert last["recycle_expected"] is True  # any RSS is above 1 KiB
    assert last["rss_end_mb"] > 0

    assert _gauge("ingestion_worker_child_tasks") == 2
    assert (_gauge("ingestion_worker_rss_bytes", point="task_end") or 0) > 0
    assert (_gauge("ingestion_worker_peak_rss_bytes", stage="task") or 0) > 0
    assert _gauge("ingestion_worker_stage_growth_bytes", stage="parse_pdf_docling") is not None
    assert (_gauge("ingestion_worker_malloc_free_bytes") or 0) >= 0
    assert (_gauge("ingestion_worker_recycles_total") or 0) - recycles_before == 2


def test_records_memory_when_the_pipeline_fails(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def failing_pipeline(_document_id: str, mem: TaskMemory | None = None) -> None:
        assert mem is not None
        mem.stage("parse_pdf_docling")
        raise ValueError("boom")

    monkeypatch.setattr(tasks, "_run_pipeline", failing_pipeline)
    monkeypatch.setenv("INGEST_WORKER_MAX_MEMORY_PER_CHILD_KB", "0")

    with caplog.at_level(logging.INFO, logger=tasks.logger.name), pytest.raises(ValueError):
        _ingest("doc-3")

    (record,) = _memory_records(caplog)
    assert set(record.__dict__["stage_peaks_mb"]) == {"parse_pdf_docling"}
    assert record.__dict__["recycle_expected"] is False, "0 turns the recycle off"


def test_instrumentation_errors_do_not_fail_the_task(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def ok_pipeline(_document_id: str, _mem: TaskMemory | None = None) -> None:
        return None

    def broken_finish(_self: TaskMemory):
        raise OSError("no /proc")

    monkeypatch.setattr(tasks, "_run_pipeline", ok_pipeline)
    monkeypatch.setattr(TaskMemory, "finish", broken_finish)

    with caplog.at_level(logging.INFO, logger=tasks.logger.name):
        _ingest("doc-4")

    assert any(r.getMessage() == "pipeline.memory_failed" for r in caplog.records)


@pytest.mark.parametrize("enabled", [True, False])
def test_malloc_trim_runs_only_when_enabled(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, enabled: bool
) -> None:
    async def fake_pipeline(_document_id: str, mem: TaskMemory | None = None) -> None:
        assert mem is not None
        mem.stage("parse_pdf_docling")

    calls: list[int] = []
    monkeypatch.setattr(tasks, "_run_pipeline", fake_pipeline)
    monkeypatch.setattr(tasks, "malloc_trim", lambda: calls.append(1) or True)
    monkeypatch.setenv("INGEST_MALLOC_TRIM", "true" if enabled else "false")

    with caplog.at_level(logging.INFO, logger=tasks.logger.name):
        _ingest("doc-trim")

    assert len(calls) == (1 if enabled else 0)
    assert _memory_records(caplog)[-1].__dict__["malloc_trimmed"] is enabled


def test_malloc_trim_precedes_the_floor_reading(monkeypatch: pytest.MonkeyPatch) -> None:
    """The floor and the peak billiard compares must both be read after the trim, or a
    document whose memory was returned still recycles."""
    order: list[str] = []

    async def fake_pipeline(_document_id: str, mem: TaskMemory | None = None) -> None:
        assert mem is not None
        mem.stage("parse_pdf_docling")

    real_finish = TaskMemory.finish
    monkeypatch.setattr(tasks, "_run_pipeline", fake_pipeline)
    monkeypatch.setattr(tasks, "malloc_trim", lambda: order.append("trim") or True)
    monkeypatch.setattr(
        TaskMemory, "finish", lambda self: (order.append("finish"), real_finish(self))[1]
    )
    monkeypatch.setenv("INGEST_MALLOC_TRIM", "true")

    _ingest("doc-order")

    assert order == ["trim", "finish"]
