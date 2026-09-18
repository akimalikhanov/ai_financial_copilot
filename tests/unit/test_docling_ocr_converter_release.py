"""The forced-OCR converter is dropped after the fallback that built it.

It is a second full pipeline, built only for the rare broken-font or scanned PDF. Holding it
keeps its models resident for the child's whole life, and that memory is live objects rather
than free heap, so the per-document malloc_trim cannot return it.
"""

from __future__ import annotations

import pytest

from src.services.ingestion import docling_parser


@pytest.fixture(autouse=True)
def _clear_converters() -> None:
    docling_parser.reset_converter()


class TestReleaseOcrConverter:
    def test_drops_the_held_converter(self) -> None:
        docling_parser._ocr_converter = object()  # type: ignore[assignment]

        assert docling_parser.release_ocr_converter() is True
        assert docling_parser._ocr_converter is None

    def test_reports_nothing_released_when_none_is_held(self) -> None:
        assert docling_parser.release_ocr_converter() is False

    def test_leaves_the_main_converter_alone(self) -> None:
        """Only the OCR pipeline goes. Dropping the main one would make every document reload
        the models it is supposed to keep warm across tasks."""
        main = object()
        docling_parser._converter = main  # type: ignore[assignment]
        docling_parser._ocr_converter = object()  # type: ignore[assignment]

        docling_parser.release_ocr_converter()

        assert docling_parser._converter is main

    def test_collects_cycles_and_empties_vram(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Dropping the name is not enough: the converter holds reference cycles, so without an
        explicit collect the models survive until the cyclic GC next runs on its own."""
        calls: list[str] = []
        monkeypatch.setattr(docling_parser.gc, "collect", lambda: calls.append("collect") or 0)
        monkeypatch.setattr(docling_parser, "_empty_cuda_cache", lambda: calls.append("cuda"))
        docling_parser._ocr_converter = object()  # type: ignore[assignment]

        docling_parser.release_ocr_converter()

        assert calls == ["collect", "cuda"]

    def test_does_nothing_when_no_converter_is_held(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The common path: a document that never needed OCR pays no collect, which on a large
        heap is not free."""
        calls: list[str] = []
        monkeypatch.setattr(docling_parser.gc, "collect", lambda: calls.append("collect") or 0)
        monkeypatch.setattr(docling_parser, "_empty_cuda_cache", lambda: calls.append("cuda"))

        docling_parser.release_ocr_converter()

        assert calls == []


class TestEmptyCudaCache:
    def test_survives_without_torch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The parser must not hard-depend on torch being importable."""
        import builtins

        real_import = builtins.__import__

        def _no_torch(name: str, *args: object, **kwargs: object) -> object:
            if name == "torch":
                raise ImportError("no torch")
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(builtins, "__import__", _no_torch)

        docling_parser._empty_cuda_cache()  # does not raise
