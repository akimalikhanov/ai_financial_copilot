"""Unit tests for docling_parser._check_status.

These build real Docling model instances rather than mocks: the whole point of the check is
that it reads a specific shape out of ConversionResult (errors carrying FailureCategory.TIMEOUT
and a page_no), so a version bump that changes that shape must fail here rather than silently
go back to indexing two-thirds of a filing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from docling.datamodel.base_models import (
    ConversionStatus,
    DoclingComponentType,
    ErrorItem,
    FailureCategory,
)
from docling.datamodel.document import ConversionResult, InputDocument

from src.services.ingestion.docling_parser import TruncatedParseError, _check_status

_PDF = Path("/tmp/does-not-need-to-exist.pdf")


def _error(category: FailureCategory, page_no: int | None) -> ErrorItem:
    return ErrorItem(
        component_type=DoclingComponentType.PIPELINE,
        module_name="ThreadedStandardPdfPipeline",
        error_message="document timeout exceeded",
        category=category,
        page_no=page_no,
    )


def _result(
    *,
    status: ConversionStatus,
    page_count: int,
    parsed_pages: int,
    errors: list[ErrorItem] | None = None,
) -> ConversionResult:
    # model_construct: a real InputDocument wants a readable file on disk, and none of that
    # validation is under test here.
    return ConversionResult.model_construct(
        input=InputDocument.model_construct(page_count=page_count),
        status=status,
        pages=[object()] * parsed_pages,
        errors=errors or [],
    )


class TestTruncatedParse:
    def test_page_loop_timeout_raises(self) -> None:
        """The P0-0 case: 668 of 1043 pages parsed, status PARTIAL_SUCCESS, and every page
        Docling abandoned is tagged TIMEOUT. This used to return the truncated result."""
        res = _result(
            status=ConversionStatus.PARTIAL_SUCCESS,
            page_count=1043,
            parsed_pages=668,
            errors=[_error(FailureCategory.TIMEOUT, n) for n in range(669, 1044)],
        )
        with pytest.raises(TruncatedParseError) as exc:
            _check_status(res, _PDF)
        msg = str(exc.value)
        assert "668 of 1043" in msg
        assert "375 pages were never reached" in msg

    def test_message_names_the_knob(self) -> None:
        """The error has to say which timeout to raise, or the operator is left guessing."""
        res = _result(
            status=ConversionStatus.PARTIAL_SUCCESS,
            page_count=10,
            parsed_pages=4,
            errors=[_error(FailureCategory.TIMEOUT, n) for n in range(5, 11)],
        )
        with pytest.raises(TruncatedParseError, match="DOCLING_DOCUMENT_TIMEOUT"):
            _check_status(res, _PDF)

    def test_unknown_page_count_falls_back_to_parsed_plus_missing(self) -> None:
        """input.page_count is 0 when the backend never reported one; the message must still
        name a total rather than 'of 0'."""
        res = _result(
            status=ConversionStatus.PARTIAL_SUCCESS,
            page_count=0,
            parsed_pages=4,
            errors=[_error(FailureCategory.TIMEOUT, n) for n in (5, 6)],
        )
        with pytest.raises(TruncatedParseError, match="4 of 6 pages"):
            _check_status(res, _PDF)


class TestNonTruncatedOutcomes:
    def test_ordinary_partial_success_is_warned_not_raised(self) -> None:
        """Pages that failed on their own merits are also absent from result.pages, so a bare
        page-count comparison would fail this document. The failure category is what separates
        them, and this asserts the distinction is actually load-bearing."""
        res = _result(
            status=ConversionStatus.PARTIAL_SUCCESS,
            page_count=10,
            parsed_pages=8,
            errors=[_error(FailureCategory.INFERENCE_FAILURE, n) for n in (3, 7)],
        )
        assert _check_status(res, _PDF) is res

    def test_success_passes_through(self) -> None:
        res = _result(status=ConversionStatus.SUCCESS, page_count=10, parsed_pages=10)
        assert _check_status(res, _PDF) is res

    def test_hard_failure_still_raises_plain_runtime_error(self) -> None:
        res = _result(status=ConversionStatus.FAILURE, page_count=10, parsed_pages=0)
        with pytest.raises(RuntimeError) as exc:
            _check_status(res, _PDF)
        assert not isinstance(exc.value, TruncatedParseError)

    def test_timeout_error_without_page_no_is_ignored(self) -> None:
        """Only per-page TIMEOUT entries mean pages are missing. A document-level timeout
        error carrying no page_no must not be read as a truncation."""
        res = _result(
            status=ConversionStatus.PARTIAL_SUCCESS,
            page_count=10,
            parsed_pages=10,
            errors=[_error(FailureCategory.TIMEOUT, None)],
        )
        assert _check_status(res, _PDF) is res


class TestStageAttribution:
    def test_stage_name_appears_in_the_error(self) -> None:
        """Every _check_status call site passes a stage (parse / scan_ocr / ocr_fallback);
        losing it would make an OCR-fallback truncation read as a plain parse truncation."""
        res = _result(
            status=ConversionStatus.PARTIAL_SUCCESS,
            page_count=10,
            parsed_pages=2,
            errors=[_error(FailureCategory.TIMEOUT, 3)],
        )
        with pytest.raises(TruncatedParseError, match="during ocr_fallback"):
            _check_status(res, _PDF, stage="ocr_fallback")
