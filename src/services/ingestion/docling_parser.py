"""Docling PDF parsing service. Wraps DocumentConverter, returns DoclingDocument + metadata."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import (
    PictureDescriptionVlmOptions,
    ThreadedPdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.threaded_standard_pdf_pipeline import ThreadedStandardPdfPipeline
from docling_core.types.doc.labels import DocItemLabel

from src.utils.config import (
    get_docling_do_ocr,
    get_docling_do_picture_description,
    get_docling_do_table_structure,
    get_docling_document_timeout,
    get_docling_generate_page_images,
    get_docling_generate_picture_images,
    get_docling_picture_vlm_model,
    get_docling_picture_vlm_prompt,
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


def _create_converter() -> DocumentConverter:
    """Create DocumentConverter with ThreadedStandardPdfPipeline and GPU/CPU auto-detection."""
    accel = AcceleratorOptions(device=AcceleratorDevice.AUTO)
    opts = ThreadedPdfPipelineOptions(
        accelerator_options=accel,
        do_ocr=get_docling_do_ocr(),
        do_table_structure=get_docling_do_table_structure(),
        do_picture_description=get_docling_do_picture_description(),
        picture_description_options=PictureDescriptionVlmOptions(
            repo_id=get_docling_picture_vlm_model(),
            prompt=get_docling_picture_vlm_prompt(),
        ),
        generate_picture_images=get_docling_generate_picture_images(),
        generate_page_images=get_docling_generate_page_images(),
        images_scale=2.0,
        document_timeout=get_docling_document_timeout(),
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


def reset_converter() -> None:
    """Call from worker_process_init to clear stale state after fork."""
    global _converter
    with _converter_lock:
        _converter = None


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
    result = converter.convert(pdf_path)
    if result.status == ConversionStatus.SUCCESS:
        pass
    elif result.status == ConversionStatus.PARTIAL_SUCCESS:
        _LOG.warning(
            "Docling conversion partial success: some pages may have failed, using extracted content"
        )
    else:
        raise RuntimeError(f"Docling conversion failed: {result.status}")

    document = result.document
    page_count = len(result.pages)
    extracted_title = _extract_title(document)
    parse_status = result.status.name.lower()  # e.g. success, partial_success

    return ParseResult(
        document=document,
        page_count=page_count,
        extracted_title=extracted_title,
        parse_status=parse_status,
        metadata={},
    )
