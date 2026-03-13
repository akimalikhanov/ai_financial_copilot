"""Docling chunking service. Converts a DoclingDocument into DB-ready chunk dictionaries."""

from __future__ import annotations

import threading
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Any, cast
from uuid import UUID

from docling_core.transforms.chunker.base import BaseChunk
from docling_core.transforms.chunker.doc_chunk import DocChunk, DocMeta
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.transforms.serializer.markdown import MarkdownParams, MarkdownTableSerializer
from docling_core.types.doc.labels import DocItemLabel
from pydantic import Field, PrivateAttr
from transformers import AutoTokenizer

from src.utils.config import (
    get_chunking_max_merge_multiplier,
    get_chunking_max_tokens,
    get_chunking_min_tokens,
    get_chunking_tokenizer_model,
)

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


class CustomHybridChunker(HybridChunker):
    """Hybrid chunker that merges undersized neighboring chunks more aggressively."""

    min_tokens: int = Field(default=20)
    max_merge_multiplier: float = Field(default=5.0)

    _pieces_by_key: dict[tuple[str, ...], list[dict[str, object]]] = PrivateAttr(
        default_factory=dict
    )

    @property
    def merge_limit(self) -> int:
        return int(min(self.max_tokens, self.min_tokens * self.max_merge_multiplier))

    def chunk(self, dl_doc: DoclingDocument, **kwargs: Any) -> Iterator[BaseChunk]:
        self._pieces_by_key.clear()

        chunks = [DocChunk.model_validate(c) for c in super().chunk(dl_doc=dl_doc, **kwargs)]

        for chunk in chunks:
            self._pieces_by_key[self._chunk_key(chunk)] = [self._piece(chunk)]

        out: list[DocChunk] = []
        i = 0

        while i < len(chunks):
            cur = chunks[i]

            if self._count_chunk_tokens(cur) >= self.min_tokens:
                out.append(cur)
                i += 1
                continue

            group = [cur]
            i += 1

            while i < len(chunks):
                nxt = chunks[i]
                if self._count_chunk_tokens(nxt) >= self.min_tokens:
                    break

                candidate = self._merge([*group, nxt])
                if self._count_chunk_tokens(candidate) > self.merge_limit:
                    break

                group.append(nxt)
                i += 1

            merged = group[0] if len(group) == 1 else self._merge(group)

            if self._count_chunk_tokens(merged) < self.min_tokens:
                if i < len(chunks):
                    candidate = self._merge([merged, chunks[i]])
                    if self._count_chunk_tokens(candidate) <= self.merge_limit:
                        chunks[i] = candidate
                        continue

                if out:
                    candidate = self._merge([out[-1], merged])
                    if self._count_chunk_tokens(candidate) <= self.merge_limit:
                        out[-1] = candidate
                        continue

            out.append(merged)

        yield from out

    def contextualize(self, chunk: BaseChunk) -> str:
        doc_chunk = DocChunk.model_validate(chunk)
        pieces = self._pieces_by_key.get(self._chunk_key(doc_chunk), [self._piece(doc_chunk)])

        rendered: list[str] = []
        for piece in pieces:
            headings = self._unique(cast(Iterable[str], piece["headings"]))
            text = cast(str, piece["text"])

            if headings:
                rendered.append("\n".join(f"[SECTION] {h}" for h in headings))
            if text:
                rendered.append(text)

        return self.delim.join(x for x in rendered if x)

    def _merge(self, chunks: list[DocChunk]) -> DocChunk:
        merged = DocChunk(
            text=self.delim.join(chunk.text for chunk in chunks if chunk.text),
            meta=DocMeta(
                doc_items=[item for chunk in chunks for item in chunk.meta.doc_items],
                headings=self._unique(h for chunk in chunks for h in (chunk.meta.headings or []))
                or None,
                origin=chunks[0].meta.origin,
            ),
        )

        pieces: list[dict[str, object]] = []
        for chunk in chunks:
            pieces.extend(self._pieces_by_key.get(self._chunk_key(chunk), [self._piece(chunk)]))

        self._pieces_by_key[self._chunk_key(merged)] = pieces
        return merged

    def _piece(self, chunk: DocChunk) -> dict[str, object]:
        return {
            "headings": self._unique(chunk.meta.headings or []),
            "text": chunk.text,
        }

    @staticmethod
    def _unique(values: Iterable[str]) -> list[str]:
        return list(dict.fromkeys(v.strip() for v in values if v and v.strip()))

    @staticmethod
    def _chunk_key(chunk: DocChunk) -> tuple[str, ...]:
        return tuple(
            ref for ref in (getattr(item, "self_ref", None) for item in chunk.meta.doc_items) if ref
        )


def _dump_model(value: Any) -> Any:
    return value.model_dump(mode="python") if hasattr(value, "model_dump") else value


def _label_to_str(item: Any) -> str | None:
    label = getattr(item, "label", None)
    if label is None:
        return None
    return getattr(label, "value", str(label))


def parse_chunk_metadata(chunk: BaseChunk) -> dict[str, Any]:
    doc_chunk = DocChunk.model_validate(chunk)
    meta = doc_chunk.meta
    provenance_rows: list[dict[str, Any]] = []
    page_set: set[int] = set()
    doc_item_refs: list[str] = []

    for item in meta.doc_items:
        item_ref = getattr(item, "self_ref", None)
        if item_ref:
            doc_item_refs.append(item_ref)

        for prov in getattr(item, "prov", []) or []:
            page_no = getattr(prov, "page_no", None)
            if isinstance(page_no, int):
                page_set.add(page_no)
            provenance_rows.append(
                {
                    "self_ref": item_ref,
                    "label": _label_to_str(item),
                    "page_no": page_no,
                    "bbox": _dump_model(getattr(prov, "bbox", None)),
                    "charspan": list(charspan)
                    if (charspan := getattr(prov, "charspan", None))
                    else None,
                }
            )

    pages = sorted(page_set)
    provenance_by_page: dict[int, list[dict[str, Any]]] = {}
    for row in provenance_rows:
        page_no = row["page_no"]
        if isinstance(page_no, int):
            provenance_by_page.setdefault(page_no, []).append(row)

    return {
        "headings": meta.headings or [],
        "origin": _dump_model(meta.origin),
        "pages": pages,
        "page_span": {
            "start": pages[0] if pages else None,
            "end": pages[-1] if pages else None,
        },
        "doc_item_refs": doc_item_refs,
        "prov": provenance_rows,
        "prov_by_page": provenance_by_page,
    }


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
    Chunk a DoclingDocument with the custom HybridChunker.

    Returns a list of dicts ready for ChunkRepository.create_many().
    Each chunk includes document_id for tracing and Qdrant payload.
    """
    tokenizer = _get_tokenizer()
    chunker = CustomHybridChunker(
        tokenizer=tokenizer,
        merge_peers=True,
        serializer_provider=AnnualReportSerializerProvider(),
        min_tokens=get_chunking_min_tokens(),
        max_merge_multiplier=get_chunking_max_merge_multiplier(),
    )
    doc_id_str = str(document_id)
    rows: list[dict[str, Any]] = []

    for i, chunk in enumerate(chunker.chunk(dl_doc=document)):
        doc_chunk = DocChunk.model_validate(chunk)
        doc_items = list(doc_chunk.meta.doc_items)
        enriched_text = chunker.contextualize(chunk=chunk)
        chunk_meta = parse_chunk_metadata(doc_chunk)
        page_span = chunk_meta["page_span"]
        labels = sorted({label for item in doc_items if (label := _label_to_str(item)) is not None})

        rows.append(
            {
                "document_id": doc_id_str,
                "chunk_index": i,
                "raw_text": chunk.text,
                "enriched_text": enriched_text,
                "heading_trail": chunk_meta["headings"] or None,
                "chunk_type": _infer_chunk_type(doc_items),
                "page_start": page_span["start"],
                "page_end": page_span["end"],
                "token_count": tokenizer.count_tokens(enriched_text),
                "provenance": chunk_meta["prov"],
                "metadata": {**chunk_meta, "labels": labels},
            }
        )

    return rows
