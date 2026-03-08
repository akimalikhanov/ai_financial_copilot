"""Docling chunking service. Converts a DoclingDocument into DB-ready chunk dictionaries."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any
from uuid import UUID

from docling_core.transforms.chunker.doc_chunk import DocChunk
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.transforms.serializer.markdown import MarkdownParams, MarkdownTableSerializer
from docling_core.types.doc.labels import DocItemLabel
from transformers import AutoTokenizer

from src.utils.config import get_chunking_max_tokens, get_chunking_tokenizer_model

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument


class AnnualReportSerializerProvider(ChunkingSerializerProvider):
    """Serialize table chunks as markdown tables with stable image placeholders."""

    def get_serializer(self, doc: Any) -> ChunkingDocSerializer:
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
            params=MarkdownParams(image_placeholder="<!-- image -->"),
        )


def _provenance_to_dict(prov: Any) -> dict[str, Any]:
    bbox = getattr(prov, "bbox", None)
    return {
        "page_no": getattr(prov, "page_no", None),
        "bbox": bbox.model_dump()
        if bbox is not None and hasattr(bbox, "model_dump")
        else (bbox.__dict__ if bbox else None),
        "charspan": getattr(prov, "charspan", None),
    }


def _extract_chunk_provenance(
    doc_items: Iterable[Any],
) -> tuple[int | None, int | None, list[dict[str, Any]]]:
    pages: list[int] = []
    out: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    for item in doc_items:
        for prov in getattr(item, "prov", None) or []:
            d = _provenance_to_dict(prov)
            key = (
                d.get("page_no"),
                json.dumps(d.get("bbox"), sort_keys=True) if d.get("bbox") is not None else None,
                tuple(d.get("charspan") or []),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(d)
            page_no = d.get("page_no")
            if isinstance(page_no, int):
                pages.append(page_no)

    if not pages:
        return None, None, out
    return min(pages), max(pages), out


def _infer_chunk_type(doc_items: Iterable[Any]) -> str:
    if any(getattr(item, "label", None) == DocItemLabel.TABLE for item in doc_items):
        return "table"
    if any(getattr(item, "label", None) == DocItemLabel.PICTURE for item in doc_items):
        return "picture"
    return "text"


_tokenizer: HuggingFaceTokenizer | None = None
_tokenizer_lock = threading.Lock()


def _get_tokenizer() -> HuggingFaceTokenizer:
    """Lazy-init singleton. Thread-safe. Must be called after fork (Celery prefork)."""
    global _tokenizer
    if _tokenizer is not None:
        return _tokenizer
    with _tokenizer_lock:
        if _tokenizer is not None:
            return _tokenizer
        hf = AutoTokenizer.from_pretrained(get_chunking_tokenizer_model())
        _tokenizer = HuggingFaceTokenizer(tokenizer=hf, max_tokens=get_chunking_max_tokens())
        return _tokenizer


def reset_tokenizer() -> None:
    """Call from worker_process_init to clear stale state after fork."""
    global _tokenizer
    with _tokenizer_lock:
        _tokenizer = None


def chunk_document(document: DoclingDocument, document_id: UUID | str) -> list[dict[str, Any]]:
    """
    Chunk a DoclingDocument with HybridChunker.

    Returns a list of dicts ready for ChunkRepository.create_many().
    Each chunk includes document_id for tracing and Qdrant payload.
    """
    tokenizer = _get_tokenizer()
    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=True,
        serializer_provider=AnnualReportSerializerProvider(),
    )
    doc_id_str = str(document_id)
    rows: list[dict[str, Any]] = []

    for i, chunk in enumerate(chunker.chunk(dl_doc=document)):
        doc_chunk = DocChunk.model_validate(chunk)
        doc_items = list(doc_chunk.meta.doc_items)
        enriched_text = chunker.contextualize(chunk=chunk)

        page_start, page_end, provenance = _extract_chunk_provenance(doc_items)
        heading_trail = list(doc_chunk.meta.headings or [])
        labels = sorted(
            {str(item.label) for item in doc_items if getattr(item, "label", None) is not None}
        )
        doc_item_refs = [
            item.self_ref for item in doc_items if getattr(item, "self_ref", None) is not None
        ]

        rows.append(
            {
                "document_id": doc_id_str,
                "chunk_index": i,
                "raw_text": chunk.text,
                "enriched_text": enriched_text,
                "heading_trail": heading_trail or None,
                "chunk_type": _infer_chunk_type(doc_items),
                "page_start": page_start,
                "page_end": page_end,
                "token_count": tokenizer.count_tokens(enriched_text),
                "provenance": provenance,
                "metadata": {"labels": labels, "doc_item_refs": doc_item_refs},
            }
        )

    return rows
