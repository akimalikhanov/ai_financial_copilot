"""Process memory readings for the ingestion worker.

Linux-only sources, each returning ``None`` (or ``False``) elsewhere so callers never branch:

* ``VmRSS`` / ``VmHWM`` from ``/proc/self/status``: current and peak resident memory.
* ``/proc/self/clear_refs``: writing ``5`` resets the peak to the *current* RSS. It resets
  ``getrusage().ru_maxrss`` too, which is what billiard's ``--max-memory-per-child`` reads
  after each task. Resetting at task end therefore makes the recycle compare the memory the
  child kept after the task, not the task's peak. That is what the next task starts on, and a
  large document that frees its memory does not cost a model reload.
* glibc ``mallinfo2()``: free bytes sitting in the allocator's heaps. Note this tracks the
  free lists, not residency — ``malloc_trim`` hands the pages back to the OS while leaving
  them listed, so RSS falls and ``fordblks`` does not. Use it to size what *might* be
  reclaimable; measure RSS to see what was.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

_STATUS = Path("/proc/self/status")
_CLEAR_REFS = Path("/proc/self/clear_refs")
_RESET_PEAK = "5"


_PROC = Path("/proc")
_CGROUP = Path("/sys/fs/cgroup")
# memory.stat keys worth exporting: working set is current - inactive_file (cadvisor's rule).
CGROUP_STAT_KEYS = ("anon", "file", "inactive_file", "shmem")


def _status_kib(key: str, status: Path | None = None) -> int | None:
    try:
        text = (status or _STATUS).read_text()
    except OSError:
        return None
    match = re.search(rf"^{key}:\s+(\d+) kB", text, re.MULTILINE)
    return int(match.group(1)) if match else None


def child_pids(pid: int | str = "self") -> list[int]:
    """Direct children of a process, from every thread's children list. Includes zombies."""
    pids: set[int] = set()
    for children in (_PROC / str(pid) / "task").glob("*/children"):
        try:
            pids.update(int(p) for p in children.read_text().split())
        except (OSError, ValueError):
            continue
    return sorted(pids)


def descendant_pids(pid: int) -> list[int]:
    """All descendants of pid (children's children and below), e.g. torch compile workers."""
    found: list[int] = []
    stack = child_pids(pid)
    while stack:
        current = stack.pop()
        found.append(current)
        stack.extend(child_pids(current))
    return sorted(found)


def pid_pss_bytes(pid: int) -> int | None:
    """Proportional set size: pages shared by N processes count 1/N in each.

    Unlike RSS, PSS adds up across processes. After a fork the parent and child share most of
    their pages, and torch.compile forks a pool of workers that share almost everything, so
    summed RSS can exceed the whole container.
    """
    try:
        text = (_PROC / str(pid) / "smaps_rollup").read_text()
    except OSError:  # gone, a zombie, or not ours
        return None
    match = re.search(r"^Pss:\s+(\d+) kB", text, re.MULTILINE)
    return int(match.group(1)) * 1024 if match else None


def cgroup_memory() -> dict[str, int]:
    """This container's cgroup v2 memory: ``current`` plus CGROUP_STAT_KEYS. Empty elsewhere."""
    out: dict[str, int] = {}
    try:
        out["current"] = int((_CGROUP / "memory.current").read_text())
        for line in (_CGROUP / "memory.stat").read_text().splitlines():
            key, _, value = line.partition(" ")
            if key in CGROUP_STAT_KEYS:
                out[key] = int(value)
    except (OSError, ValueError):
        return {}
    return out


def rss_bytes() -> int | None:
    kib = _status_kib("VmRSS")
    return None if kib is None else kib * 1024


def peak_bytes() -> int | None:
    kib = _status_kib("VmHWM")
    return None if kib is None else kib * 1024


def reset_peak() -> bool:
    try:
        _CLEAR_REFS.write_text(_RESET_PEAK)
    except OSError:
        return False
    return True


class _MallInfo2(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_size_t)
        for name in (
            "arena",
            "ordblks",
            "smblks",
            "hblks",
            "hblkhd",
            "usmblks",
            "fsmblks",
            "uordblks",
            "fordblks",
            "keepcost",
        )
    ]


@dataclass(frozen=True)
class MallocStats:
    arena: int  # heap bytes obtained with brk/sbrk and thread-arena mmaps
    hblkhd: int  # bytes in large blocks mmapped one by one
    fordblks: int  # free bytes inside the heaps, held by glibc


def _load_libc_fn(name: str):
    lib = ctypes.util.find_library("c")
    if lib is None:
        return None
    try:
        return getattr(ctypes.CDLL(lib), name)
    except (OSError, AttributeError):  # not glibc, or too old for this symbol
        return None


def _load_mallinfo2():
    fn = _load_libc_fn("mallinfo2")
    if fn is None:
        return None
    fn.restype = _MallInfo2
    fn.argtypes = []
    return fn


def _load_malloc_trim():
    fn = _load_libc_fn("malloc_trim")
    if fn is None:
        return None
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_size_t]
    return fn


_mallinfo2 = _load_mallinfo2()
_malloc_trim = _load_malloc_trim()


def malloc_stats() -> MallocStats | None:
    if _mallinfo2 is None:
        return None
    info = _mallinfo2()
    return MallocStats(arena=info.arena, hblkhd=info.hblkhd, fordblks=info.fordblks)


def malloc_trim(pad: int = 0) -> bool:
    """Ask glibc to return free heap memory to the OS. True when it released anything.

    Measure the effect on RSS, not on ``fordblks``: the pages are madvised away but stay on
    the free lists, so a trim that returns hundreds of MB leaves ``fordblks`` unchanged. Only
    page-aligned free spans can go, so a heap with live allocations scattered through it can
    report a large free total and release little of it.

    Costs O(free blocks) across every arena, and glibc allows up to 8x the *host's* CPU count
    of them, so the call is not free on a large fragmented heap. Callers on a latency path
    should measure it rather than assume.
    """
    if _malloc_trim is None:
        return False
    return bool(_malloc_trim(pad))


def to_mb(value: int | None) -> float | None:
    return None if value is None else round(value / 1e6, 1)


@dataclass
class TaskMemory:
    """Floors, per-stage peaks and per-stage growth for one task.

    ``stage()`` closes the previous stage and resets the peak for the next one. A stage's peak
    includes whatever earlier stages left behind; its growth (peak minus the RSS it started at)
    is the memory the stage itself needed.
    ``finish()`` closes the last stage, reads the floor, and resets once more so the recycle
    check that follows sees the floor.
    """

    child_tasks: int
    rss_start: int | None = None
    malloc_start: MallocStats | None = None
    stage_peaks: dict[str, int] = field(default_factory=dict)
    stage_growth: dict[str, int] = field(default_factory=dict)
    page_count: int | None = None
    _stage: str | None = None
    _stage_rss_start: int | None = None

    @classmethod
    def start(cls, child_tasks: int) -> TaskMemory:
        mem = cls(child_tasks=child_tasks, rss_start=rss_bytes(), malloc_start=malloc_stats())
        reset_peak()
        return mem

    def stage(self, name: str) -> None:
        self._close_stage()
        self._stage = name
        self._stage_rss_start = rss_bytes()
        reset_peak()

    def _close_stage(self) -> None:
        if self._stage is None:
            return
        peak = peak_bytes()
        if peak is not None:
            # A stage name can repeat (finalize on the no-chunks path): keep the larger.
            name = self._stage
            self.stage_peaks[name] = max(peak, self.stage_peaks.get(name, 0))
            if self._stage_rss_start is not None:
                growth = max(peak - self._stage_rss_start, 0)
                self.stage_growth[name] = max(growth, self.stage_growth.get(name, 0))
        self._stage = None

    @property
    def task_peak(self) -> int | None:
        return max(self.stage_peaks.values()) if self.stage_peaks else None

    def finish(self) -> TaskMemoryReport:
        self._close_stage()
        report = TaskMemoryReport(task=self, rss_end=rss_bytes(), malloc_end=malloc_stats())
        reset_peak()
        return report


@dataclass(frozen=True)
class TaskMemoryReport:
    task: TaskMemory
    rss_end: int | None
    malloc_end: MallocStats | None

    def log_fields(self) -> dict[str, object]:
        t = self.task
        return {
            "child_pid": os.getpid(),
            "child_tasks": t.child_tasks,
            "page_count": t.page_count,
            "rss_start_mb": to_mb(t.rss_start),
            "rss_end_mb": to_mb(self.rss_end),
            "peak_mb": to_mb(t.task_peak),
            "stage_peaks_mb": {k: to_mb(v) for k, v in t.stage_peaks.items()},
            "stage_growth_mb": {k: to_mb(v) for k, v in t.stage_growth.items()},
            "malloc_free_start_mb": to_mb(t.malloc_start.fordblks if t.malloc_start else None),
            "malloc_free_end_mb": to_mb(self.malloc_end.fordblks if self.malloc_end else None),
            "malloc_arena_end_mb": to_mb(self.malloc_end.arena if self.malloc_end else None),
            "malloc_mmapped_end_mb": to_mb(self.malloc_end.hblkhd if self.malloc_end else None),
        }
