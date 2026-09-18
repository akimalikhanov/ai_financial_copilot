"""Process memory helpers."""

from __future__ import annotations

import gc
import os
import resource
import sys
import time
from pathlib import Path

import pytest

from src.observability import process_memory as pm

linux_only = pytest.mark.skipif(sys.platform != "linux", reason="reads /proc")

MB = 1024 * 1024


def _touch(n_bytes: int) -> bytearray:
    buf = bytearray(n_bytes)
    buf[::4096] = b"x" * len(buf[::4096])
    return buf


def _base() -> int:
    rss = pm.rss_bytes()
    assert rss is not None
    return rss


def _in_child(fn) -> None:
    """Run fn in a forked child, the way a prefork pool child starts, and fail on its failure."""
    pid = os.fork()
    if pid == 0:  # pragma: no cover — reported through the exit code
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001
            print(f"child failed: {exc!r}", file=sys.stderr)
            os._exit(1)
        os._exit(0)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0


@linux_only
class TestReadings:
    def test_rss_and_peak_are_positive(self) -> None:
        rss, peak = pm.rss_bytes(), pm.peak_bytes()
        assert rss is not None and peak is not None
        assert 0 < rss <= peak

    def test_malloc_stats_on_glibc(self) -> None:
        stats = pm.malloc_stats()
        assert stats is not None
        assert stats.arena > 0 and stats.fordblks >= 0

    def test_malloc_trim_returns_freed_heap_to_the_os(self) -> None:
        """RSS is the metric, not fordblks.

        Trimming madvises the free pages away but leaves them on the heap's free lists, so
        fordblks stays put while RSS drops by the whole amount. Asserting on fordblks would
        report a working trim as a no-op.
        """

        def body() -> None:
            base = _base()
            # Small blocks, so they come from the heap rather than being mmapped one by one:
            # an mmapped block is returned on free anyway and would trim nothing.
            blocks = [bytearray(64 * 1024) for _ in range(4000)]
            for b in blocks:
                b[0] = 1
            del blocks
            gc.collect()
            before_rss = pm.rss_bytes()
            before = pm.malloc_stats()
            assert before_rss is not None and before is not None
            assert before_rss > base + 200 * MB, "freed, but still resident"
            assert before.fordblks > 200 * MB

            assert pm.malloc_trim() is True

            after_rss = pm.rss_bytes()
            assert after_rss is not None
            assert after_rss < before_rss - 200 * MB

        _in_child(body)

    def test_reset_sets_the_peak_to_current_rss_and_ru_maxrss_follows(self) -> None:
        def body() -> None:
            base = _base()
            big = _touch(300 * MB)
            del big
            before = pm.peak_bytes()
            assert before is not None and before > base + 250 * MB
            assert pm.reset_peak()
            after, rss = pm.peak_bytes(), pm.rss_bytes()
            assert after is not None and rss is not None
            assert after < base + 100 * MB, "peak dropped to about the current RSS"
            assert after >= rss - 4 * MB
            # billiard's --max-memory-per-child reads this, in KiB.
            maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            assert maxrss < base + 100 * MB

        _in_child(body)


class TestFallbacks:
    def test_no_proc(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(pm, "_STATUS", tmp_path / "missing")
        monkeypatch.setattr(pm, "_CLEAR_REFS", tmp_path / "no-dir" / "clear_refs")
        assert pm.rss_bytes() is None
        assert pm.peak_bytes() is None
        assert pm.reset_peak() is False

    def test_no_mallinfo2(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pm, "_mallinfo2", None)
        assert pm.malloc_stats() is None

    def test_no_malloc_trim(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pm, "_malloc_trim", None)
        assert pm.malloc_trim() is False

    def test_report_without_readings(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(pm, "_STATUS", tmp_path / "missing")
        monkeypatch.setattr(pm, "_CLEAR_REFS", tmp_path / "no-dir" / "clear_refs")
        monkeypatch.setattr(pm, "_mallinfo2", None)
        mem = pm.TaskMemory.start(child_tasks=1)
        mem.stage("parse")
        report = mem.finish()
        fields = report.log_fields()
        assert fields["rss_end_mb"] is None
        assert fields["peak_mb"] is None
        assert fields["stage_peaks_mb"] == {}
        assert fields["malloc_free_end_mb"] is None


@linux_only
class TestTaskMemory:
    def test_stage_peaks_are_separate(self) -> None:
        def body() -> None:
            base = _base()
            mem = pm.TaskMemory.start(child_tasks=1)
            mem.stage("parse")
            big = _touch(300 * MB)
            del big
            mem.stage("chunk")
            small = _touch(20 * MB)
            del small
            report = mem.finish()
            assert mem.stage_peaks["parse"] > base + 250 * MB
            assert mem.stage_peaks["chunk"] < base + 100 * MB, "parse's peak stayed in parse"
            assert mem.stage_growth["parse"] > 250 * MB
            assert mem.stage_growth["chunk"] < 60 * MB
            assert mem.task_peak == mem.stage_peaks["parse"]
            fields = report.log_fields()
            assert fields["child_tasks"] == 1
            assert set(fields["stage_peaks_mb"]) == {"parse", "chunk"}  # type: ignore[arg-type]

        _in_child(body)

    def test_growth_excludes_what_earlier_stages_kept(self) -> None:
        def body() -> None:
            base = _base()
            mem = pm.TaskMemory.start(child_tasks=1)
            mem.stage("parse")
            kept = _touch(300 * MB)
            mem.stage("chunk")
            extra = _touch(50 * MB)
            del extra
            mem.finish()
            assert mem.stage_peaks["chunk"] > base + 300 * MB, "the peak includes what parse kept"
            assert 30 * MB < mem.stage_growth["chunk"] < 100 * MB, (
                "growth counts only chunk's 50 MB"
            )
            fields = mem.finish().log_fields()
            assert set(fields["stage_growth_mb"]) == {"parse", "chunk"}  # type: ignore[arg-type]
            assert len(kept) == 300 * MB

        _in_child(body)

    def test_repeated_stage_keeps_the_larger_peak(self) -> None:
        def body() -> None:
            base = _base()
            mem = pm.TaskMemory.start(child_tasks=1)
            mem.stage("finalize")
            big = _touch(200 * MB)
            del big
            mem.stage("finalize")
            mem.finish()
            assert mem.stage_peaks["finalize"] > base + 150 * MB

        _in_child(body)

    def test_finish_leaves_the_floor_for_the_recycle_check(self) -> None:
        """billiard compares the threshold with what the task kept, not with its peak.

        A large transient peak is not seen; memory the task kept is.
        """

        def body() -> None:
            base = _base()
            mem = pm.TaskMemory.start(child_tasks=1)
            mem.stage("parse")
            transient = _touch(400 * MB)
            kept = _touch(250 * MB)
            del transient
            report = mem.finish()
            maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            assert mem.task_peak is not None and mem.task_peak > base + 600 * MB
            assert report.rss_end is not None and report.rss_end > base + 200 * MB
            assert base + 200 * MB < maxrss < base + 400 * MB, "kept 250 MB, not the 650 MB peak"
            assert len(kept) == 250 * MB

        _in_child(body)


@linux_only
def test_pss_splits_pages_shared_after_fork() -> None:
    shared = _touch(200 * MB)
    read_fd, write_fd = os.pipe()
    child = os.fork()
    if child == 0:  # pragma: no cover
        os.read(read_fd, 1)
        os._exit(0)
    try:
        parent_rss = pm.rss_bytes()
        parent_pss = pm.pid_pss_bytes(os.getpid())
        child_pss = pm.pid_pss_bytes(child)
        assert parent_rss is not None and parent_pss is not None and child_pss is not None
        assert parent_pss < parent_rss - 80 * MB, "the shared 200 MB is split, not counted twice"
        assert pm.pid_pss_bytes(2**22 + 12345) is None
    finally:
        os.write(write_fd, b"x")
        os.waitpid(child, 0)
    assert len(shared) == 200 * MB


class TestCgroupMemory:
    def test_reads_current_and_selected_stat_keys(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        (tmp_path / "memory.current").write_text("9204674560\n")
        (tmp_path / "memory.stat").write_text(
            "anon 7000000000\nfile 2000000000\nkernel 50000000\n"
            "shmem 1000\ninactive_file 1500000000\n"
        )
        monkeypatch.setattr(pm, "_CGROUP", tmp_path)
        assert pm.cgroup_memory() == {
            "current": 9204674560,
            "anon": 7000000000,
            "file": 2000000000,
            "shmem": 1000,
            "inactive_file": 1500000000,
        }

    def test_empty_outside_a_memory_cgroup(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(pm, "_CGROUP", tmp_path)
        assert pm.cgroup_memory() == {}


@linux_only
def test_process_tree_collector(monkeypatch: pytest.MonkeyPatch) -> None:
    """A live child with its own child, plus a zombie that must not be counted."""
    from src.observability.worker_metrics import ProcessTreeMemoryCollector

    monkeypatch.setattr("src.observability.worker_metrics.cgroup_memory", lambda: {"current": 5})
    zombie = os.fork()
    if zombie == 0:  # pragma: no cover
        os._exit(0)
    read_fd, write_fd = os.pipe()
    ready_r, ready_w = os.pipe()
    child = os.fork()
    if child == 0:  # pragma: no cover
        os.close(write_fd)
        grandchild = os.fork()
        if grandchild == 0:
            os.read(read_fd, 1)
            os._exit(0)
        os.write(ready_w, b"r")
        os.read(read_fd, 1)
        os.waitpid(grandchild, 0)
        os._exit(0)
    os.close(read_fd)
    os.read(ready_r, 1)
    time.sleep(0.2)  # let the zombie exit (it is not reaped until the end)
    try:
        assert {zombie, child} <= set(pm.child_pids())
        assert len(pm.descendant_pids(child)) == 1
        samples = {
            (m.name, tuple(s.labels.items())): s.value
            for m in ProcessTreeMemoryCollector().collect()
            for s in m.samples
        }
    finally:
        os.write(write_fd, b"xx")
        os.waitpid(child, 0)
        os.waitpid(zombie, 0)
    assert samples[("worker_process_pss_bytes", (("role", "parent"),))] > 0
    assert samples[("worker_process_pss_bytes", (("role", "children"),))] > 0
    assert samples[("worker_process_pss_bytes", (("role", "descendants"),))] > 0
    assert samples[("worker_pool_children", ())] == 1, "the zombie is not a live child"
    assert samples[("worker_cgroup_memory_bytes", (("kind", "current"),))] == 5
