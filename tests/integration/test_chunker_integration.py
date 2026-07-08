"""Integration test for CustomHybridChunker against a real Docling document + real HF tokenizer.

Unlike tests/unit/test_chunker.py (which stubs the tokenizer and monkeypatches the base
HybridChunker.chunk to hand-built DocChunks), this exercises the real docling chunking
pass and a real HuggingFace tokenizer's BPE token counts. Downloads/caches a small
tokenizer on first run, hence integration-tier, not unit-tier.

Table/picture chunk-type inference is not covered here (no suitable small fixture with
those doc items) — see tests/unit/test_chunker.py::TestInferChunkType for that coverage.
"""

from __future__ import annotations

import pytest
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel
from transformers import AutoTokenizer

from src.services.ingestion.chunker import AnnualReportSerializerProvider, CustomHybridChunker
from src.utils.config import get_chunking_max_tokens, get_chunking_tokenizer_model

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def real_tokenizer() -> HuggingFaceTokenizer:
    """Real tokenizer from CHUNKING_TOKENIZER_MODEL (downloads tokenizer files only, ~20MB)."""
    hf = AutoTokenizer.from_pretrained(get_chunking_tokenizer_model())
    return HuggingFaceTokenizer(tokenizer=hf, max_tokens=get_chunking_max_tokens())


def _make_document() -> DoclingDocument:
    doc = DoclingDocument(name="integration-test-doc")
    doc.add_heading(text="Financial Highlights")
    doc.add_text(label=DocItemLabel.TEXT, text="Revenue increased.")  # undersized
    doc.add_text(label=DocItemLabel.TEXT, text="Costs were stable.")  # undersized
    doc.add_heading(text="Full Report")
    doc.add_text(
        label=DocItemLabel.TEXT,
        text=" ".join(f"word{i}" for i in range(300)),  # oversized on its own
    )
    return doc


class TestRealChunking:
    def test_undersized_paragraphs_merged_with_real_tokenizer(
        self, real_tokenizer: HuggingFaceTokenizer
    ) -> None:
        chunker = CustomHybridChunker(
            tokenizer=real_tokenizer,
            merge_peers=True,
            serializer_provider=AnnualReportSerializerProvider(),
            min_tokens=10,
            max_merge_multiplier=10.0,
        )
        document = _make_document()

        chunks = list(chunker.chunk(dl_doc=document))

        assert len(chunks) >= 1
        merged_texts = " ".join(chunker.contextualize(chunk=c) for c in chunks)
        assert "Revenue increased." in merged_texts
        assert "Costs were stable." in merged_texts

    def test_chunks_respect_max_tokens_with_real_token_counts(
        self, real_tokenizer: HuggingFaceTokenizer
    ) -> None:
        chunker = CustomHybridChunker(
            tokenizer=real_tokenizer,
            merge_peers=True,
            serializer_provider=AnnualReportSerializerProvider(),
            min_tokens=10,
            max_merge_multiplier=5.0,
        )
        document = _make_document()

        for chunk in chunker.chunk(dl_doc=document):
            text = chunker.contextualize(chunk=chunk)
            assert real_tokenizer.count_tokens(text) <= chunker.max_tokens

    def test_merge_limit_computed_from_real_max_tokens(
        self, real_tokenizer: HuggingFaceTokenizer
    ) -> None:
        chunker = CustomHybridChunker(
            tokenizer=real_tokenizer,
            merge_peers=True,
            serializer_provider=AnnualReportSerializerProvider(),
            min_tokens=10,
            max_merge_multiplier=2.0,
        )
        assert chunker.merge_limit == min(chunker.max_tokens, 20)
