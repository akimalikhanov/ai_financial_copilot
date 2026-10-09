"""Guardrails around the parse (findings doc §2 "Applies regardless of which option is chosen").

A page-count ceiling checked before any model runs, and an OCR re-convert that neither doubles
peak memory nor runs at all on a document too large to afford it.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pypdfium2 as pdfium
import pytest

from src.services.ingestion import docling_parser, tasks


@pytest.fixture
def pdf_3_pages(tmp_path: Path) -> Path:
    pdf = pdfium.PdfDocument.new()
    for _ in range(3):
        pdf.new_page(200, 300)
    path = tmp_path / "three.pdf"
    pdf.save(str(path))
    pdf.close()
    return path


class TestProbePageCount:
    def test_reads_the_count_from_the_pdf(self, pdf_3_pages: Path) -> None:
        assert docling_parser.probe_page_count(pdf_3_pages) == 3

    def test_unreadable_file_returns_none(self, tmp_path: Path) -> None:
        """None means "cannot tell", which must let the document through to the parse."""
        broken = tmp_path / "broken.pdf"
        broken.write_bytes(b"not a pdf")
        assert docling_parser.probe_page_count(broken) is None


class TestPageLimit:
    def test_document_over_the_limit_fails_cleanly(
        self, pdf_3_pages: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A clean failure, rather than an OOM kill that burns every attempt on restarts."""
        monkeypatch.setattr(tasks, "get_ingest_max_pages", lambda: 2)

        with pytest.raises(RuntimeError, match="3 pages, above the 2-page"):
            tasks._enforce_page_limit(pdf_3_pages)

    def test_document_at_the_limit_passes(
        self, pdf_3_pages: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(tasks, "get_ingest_max_pages", lambda: 3)
        tasks._enforce_page_limit(pdf_3_pages)

    def test_zero_disables_the_guardrail(
        self, pdf_3_pages: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(tasks, "get_ingest_max_pages", lambda: 0)
        tasks._enforce_page_limit(pdf_3_pages)

    def test_unreadable_pdf_is_let_through(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(tasks, "get_ingest_max_pages", lambda: 1)
        broken = tmp_path / "broken.pdf"
        broken.write_bytes(b"not a pdf")
        tasks._enforce_page_limit(broken)

    def test_the_check_runs_before_the_parse(self) -> None:
        source = inspect.getsource(tasks._run_pipeline)
        assert source.index("_enforce_page_limit") < source.index('_log_stage("parse_pdf_docling")')


class TestOcrRetryCap:
    def test_small_document_still_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(docling_parser, "get_docling_ocr_max_pages", lambda: 300)
        assert docling_parser._ocr_retry_allowed(120, "ocr_fallback") is True

    def test_large_document_skips_the_reconvert(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A second full parse at ~4.3s/page, inside the budget the first one already spent."""
        monkeypatch.setattr(docling_parser, "get_docling_ocr_max_pages", lambda: 300)
        assert docling_parser._ocr_retry_allowed(1043, "ocr_fallback") is False

    def test_zero_disables_the_cap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(docling_parser, "get_docling_ocr_max_pages", lambda: 0)
        assert docling_parser._ocr_retry_allowed(5000, "scan_ocr") is True

    def test_skipped_fallback_does_not_report_plain_success(self) -> None:
        """Garbled text that is too large to re-OCR is still garbled: indexing it under
        "success" would hide it from the UI badge entirely."""
        source = inspect.getsource(docling_parser.parse)
        skip_branch = source.split("if garbled and not _ocr_retry_allowed")[1].split("elif")[0]
        assert 'parse_status = "garbled_text"' in skip_branch

    def test_first_conversion_is_released_before_the_reconvert(self) -> None:
        """Both results alive at once is 2x the retained page set plus the EasyOCR models —
        the exact peak the memory fix exists to avoid."""
        source = inspect.getsource(docling_parser.parse)
        for stage in ("scan_ocr", "ocr_fallback"):
            before = source.split(f'stage="{stage}"')[0].rsplit("_check_status", 1)[0][-400:]
            assert "del result" in before, stage
            # del alone leaves any reference cycles to the cyclic GC, which may not run
            # before the OCR parse starts allocating.
            assert before.index("del result") < before.index("gc.collect()"), stage
