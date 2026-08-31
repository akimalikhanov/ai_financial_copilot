"""Docling PDF parsing service. Wraps DocumentConverter, returns DoclingDocument + metadata."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    PictureDescriptionVlmOptions,
    ThreadedPdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.threaded_standard_pdf_pipeline import ThreadedStandardPdfPipeline
from docling_core.types.doc.labels import DocItemLabel

from src.services.ingestion import text_quality
from src.utils.config import (
    get_docling_artifacts_path,
    get_docling_device,
    get_docling_do_ocr,
    get_docling_do_picture_classification,
    get_docling_do_picture_description,
    get_docling_do_table_structure,
    get_docling_document_timeout,
    get_docling_generate_page_images,
    get_docling_generate_picture_images,
    get_docling_images_scale,
    get_docling_ocr_document_timeout,
    get_docling_ocr_fallback_enabled,
    get_docling_ocr_use_gpu,
    get_docling_picture_vlm_model,
    get_docling_picture_vlm_prompt,
    get_docling_text_quality_threshold,
)

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument

_LOG = logging.getLogger(__name__)


@dataclass
class ParseResult:
    """Result of parsing a PDF with Docling."""

    document: DoclingDocument
    page_count: int
    extracted_title: str | None
    parse_status: str  # Docling ConversionStatus as string: success, partial_success, etc.
    metadata: dict


def _create_converter(*, force_ocr: bool = False) -> DocumentConverter:
    """Create DocumentConverter with ThreadedStandardPdfPipeline and GPU/CPU auto-detection.

    `force_ocr` builds the fallback converter used when a parse comes back garbled: it OCRs
    every page rather than only the regions without a text layer, which is the only way to get
    past a PDF whose text layer exists but decodes to nonsense.
    """
    accel = AcceleratorOptions(device=get_docling_device(), cuda_use_flash_attention2=False)
    opts = ThreadedPdfPipelineOptions(
        accelerator_options=accel,
        do_ocr=True if force_ocr else get_docling_do_ocr(),
        ocr_options=EasyOcrOptions(
            lang=["en"],
            use_gpu=get_docling_ocr_use_gpu(),
            force_full_page_ocr=force_ocr,
        ),
        do_table_structure=get_docling_do_table_structure(),
        do_picture_description=get_docling_do_picture_description(),
        picture_description_options=PictureDescriptionVlmOptions(
            repo_id=get_docling_picture_vlm_model(),
            prompt=get_docling_picture_vlm_prompt(),
        ),
        do_picture_classification=get_docling_do_picture_classification(),
        generate_picture_images=get_docling_generate_picture_images(),
        generate_page_images=get_docling_generate_page_images(),
        images_scale=get_docling_images_scale(),
        document_timeout=(
            get_docling_ocr_document_timeout() if force_ocr else get_docling_document_timeout()
        ),
        artifacts_path=get_docling_artifacts_path(),
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=ThreadedStandardPdfPipeline,
                pipeline_options=opts,
            )
        }
    )


_converter: DocumentConverter | None = None
_ocr_converter: DocumentConverter | None = None
_converter_lock = threading.Lock()


def _get_converter() -> DocumentConverter:
    """Lazy-init singleton. Thread-safe. Must be called after fork (Celery prefork)."""
    global _converter
    if _converter is not None:
        return _converter
    with _converter_lock:
        if _converter is not None:
            return _converter
        converter = _create_converter()
        converter.initialize_pipeline(InputFormat.PDF)
        _converter = converter
        return _converter


def _get_ocr_converter() -> DocumentConverter:
    """Lazy-init singleton for the forced-OCR fallback. Built only when first needed, so the
    EasyOCR models are not loaded on workers that never meet a broken-font PDF."""
    global _ocr_converter
    if _ocr_converter is not None:
        return _ocr_converter
    with _converter_lock:
        if _ocr_converter is not None:
            return _ocr_converter
        converter = _create_converter(force_ocr=True)
        converter.initialize_pipeline(InputFormat.PDF)
        _ocr_converter = converter
        return _ocr_converter


def reset_converter() -> None:
    """Call from worker_process_init to clear stale state after fork."""
    global _converter, _ocr_converter
    with _converter_lock:
        _converter = None
        _ocr_converter = None


def _check_status(result: ConversionResult, pdf_path: Path, *, stage: str = "parse"):
    """Raise on a failed conversion; warn and continue on a partial one."""
    if result.status == ConversionStatus.PARTIAL_SUCCESS:
        _LOG.warning(
            "docling.partial_success",
            extra={
                "pdf_path": str(pdf_path),
                "stage": stage,
                "page_count": len(result.pages),
                "errors": [str(e.error_message) for e in result.errors],
            },
        )
    elif result.status != ConversionStatus.SUCCESS:
        raise RuntimeError(f"Docling conversion failed during {stage}: {result.status}")
    return result


def _extract_title(document: DoclingDocument) -> str | None:
    """Extract title from first TitleItem in document texts."""
    for item in document.texts:
        if getattr(item, "label", None) == DocItemLabel.TITLE:
            text = getattr(item, "text", None)
            if text and (s := str(text).strip()):
                return s
    return None


def parse(pdf_path: Path) -> ParseResult:
    """
    Parse PDF with Docling. Returns DoclingDocument and extracted metadata.
    Raises RuntimeError if conversion fails.
    """
    converter = _get_converter()
    result = _check_status(converter.convert(pdf_path), pdf_path)
    parse_status = result.status.name.lower()  # e.g. success, partial_success

    # A broken font encoding still parses as SUCCESS, so the status alone will not catch it.
    if get_docling_ocr_fallback_enabled():
        threshold = get_docling_text_quality_threshold()
        sample = text_quality.sample_document_text(result.document)
        garbled, ratio = text_quality.assess(sample, threshold=threshold)
        if garbled:
            _LOG.warning(
                "docling.garbled_text_retrying_with_ocr",
                extra={"pdf_path": str(pdf_path), "stopword_ratio": round(ratio, 5)},
            )
            result = _check_status(
                _get_ocr_converter().convert(pdf_path), pdf_path, stage="ocr_fallback"
            )
            still_garbled, retry_ratio = text_quality.assess(
                text_quality.sample_document_text(result.document), threshold=threshold
            )
            # Indexed either way, but never reported as plain "success": Phase 6's UI badge
            # fires on anything else, warning that the text is OCR-derived, unreadable, or
            # short some pages. A partial OCR pass must not be laundered into a clean status.
            retry_status = result.status.name.lower()
            if still_garbled:
                parse_status = "garbled_text"
            elif retry_status != "success":
                parse_status = f"{retry_status}_ocr_fallback"
            else:
                parse_status = "success_ocr_fallback"
            _LOG.warning(
                "docling.ocr_fallback_complete",
                extra={
                    "pdf_path": str(pdf_path),
                    "stopword_ratio_before": round(ratio, 5),
                    "stopword_ratio_after": round(retry_ratio, 5),
                    "parse_status": parse_status,
                },
            )

    document = result.document
    page_count = len(result.pages)
    extracted_title = _extract_title(document)

    return ParseResult(
        document=document,
        page_count=page_count,
        extracted_title=extracted_title,
        parse_status=parse_status,
        metadata={},
    )
