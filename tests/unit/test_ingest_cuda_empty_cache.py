"""The VRAM counterpart to the malloc_trim teardown.

Torch's caching allocator keeps every block it grabbed for parse-time intermediates, so without
an explicit empty_cache() a worker's VRAM floor steps up on its first document and stays there
for the life of the child. gc.collect() does not help: it returns blocks to torch's own pool,
not to the driver.
"""

from __future__ import annotations

import pytest

from src.services.ingestion import docling_parser
from src.utils.config import get_ingest_cuda_empty_cache_enabled


class TestConfigFlag:
    def test_defaults_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unlike INGEST_MALLOC_TRIM this defaults true: the retention it fixes is a plain
        defect, and releasing a cache costs milliseconds against a minutes-long parse."""
        monkeypatch.delenv("INGEST_CUDA_EMPTY_CACHE", raising=False)

        assert get_ingest_cuda_empty_cache_enabled() is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", "FALSE", " off "])
    def test_can_be_disabled(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        monkeypatch.setenv("INGEST_CUDA_EMPTY_CACHE", value)

        assert get_ingest_cuda_empty_cache_enabled() is False

    @pytest.mark.parametrize("value", ["1", "true", "yes", "on", "TRUE"])
    def test_truthy_spellings(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        monkeypatch.setenv("INGEST_CUDA_EMPTY_CACHE", value)

        assert get_ingest_cuda_empty_cache_enabled() is True


class TestEmptyCudaCache:
    def test_reports_false_without_torch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The parser must not hard-depend on torch, and the caller logs what actually ran."""
        import builtins

        real_import = builtins.__import__

        def _no_torch(name: str, *args: object, **kwargs: object) -> object:
            if name == "torch":
                raise ImportError("no torch")
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(builtins, "__import__", _no_torch)

        assert docling_parser.empty_cuda_cache() is False

    def test_reports_false_without_a_gpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CPU-only workers are a supported configuration, not an error."""
        calls: list[str] = []
        fake = _FakeTorch(available=False, on_empty=lambda: calls.append("emptied"))
        monkeypatch.setitem(__import__("sys").modules, "torch", fake)

        assert docling_parser.empty_cuda_cache() is False
        assert calls == []

    def test_empties_and_reports_true_on_a_gpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[str] = []
        fake = _FakeTorch(available=True, on_empty=lambda: calls.append("emptied"))
        monkeypatch.setitem(__import__("sys").modules, "torch", fake)

        assert docling_parser.empty_cuda_cache() is True
        assert calls == ["emptied"]

    def test_swallows_a_driver_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A failed release must never fail the document that has already been ingested."""

        def _boom() -> None:
            raise RuntimeError("CUDA driver is shutting down")

        fake = _FakeTorch(available=True, on_empty=_boom)
        monkeypatch.setitem(__import__("sys").modules, "torch", fake)

        assert docling_parser.empty_cuda_cache() is False


class _FakeTorch:
    def __init__(self, *, available: bool, on_empty) -> None:  # noqa: ANN001 — test double
        self.cuda = _FakeCuda(available=available, on_empty=on_empty)


class _FakeCuda:
    def __init__(self, *, available: bool, on_empty) -> None:  # noqa: ANN001 — test double
        self._available = available
        self._on_empty = on_empty

    def is_available(self) -> bool:
        return self._available

    def empty_cache(self) -> None:
        self._on_empty()
