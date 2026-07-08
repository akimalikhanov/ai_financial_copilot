"""Unit tests for CustomHybridChunker's merge-policy logic.

Uses a stub tokenizer (word-count based) instead of a real HF AutoTokenizer to
keep these tests fast/offline — see docs/stages/stage_16a_tests.md section 6.
`HybridChunker.chunk` (the base-class docling chunking pass) is monkeypatched to
yield hand-built DocChunks directly, so only the merge-loop logic in
CustomHybridChunker.chunk is under test, not real Docling document parsing.
"""

from __future__ import annotations

from docling_core.transforms.chunker.doc_chunk import DocChunk, DocMeta
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from docling_core.types.doc.document import TextItem
from docling_core.types.doc.labels import DocItemLabel

from src.services.ingestion.chunker import AnnualReportSerializerProvider, CustomHybridChunker


class StubTokenizer(BaseTokenizer):
    """Minimal tokenizer stub: token count = word count."""

    max_tokens: int = 1000

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def get_max_tokens(self) -> int:
        return self.max_tokens

    def get_tokenizer(self) -> StubTokenizer:
        return self


def _doc_chunk(text: str, ref: str, heading: str = "Section") -> DocChunk:
    item = TextItem(self_ref=ref, label=DocItemLabel.TEXT, orig=text, text=text)
    meta = DocMeta(doc_items=[item], headings=[heading], origin=None)
    return DocChunk(text=text, meta=meta)


def _make_chunker(
    *, min_tokens: int, max_merge_multiplier: float, max_tokens: int = 1000
) -> CustomHybridChunker:
    return CustomHybridChunker(
        tokenizer=StubTokenizer(max_tokens=max_tokens),
        merge_peers=True,
        serializer_provider=AnnualReportSerializerProvider(),
        min_tokens=min_tokens,
        max_merge_multiplier=max_merge_multiplier,
    )


def _run_chunk(
    chunker: CustomHybridChunker, docling_chunks: list[DocChunk], monkeypatch
) -> list[DocChunk]:
    monkeypatch.setattr(HybridChunker, "chunk", lambda *_a, **_kw: iter(docling_chunks))
    return [DocChunk.model_validate(c) for c in chunker.chunk(dl_doc=object())]  # type: ignore[arg-type]


class TestMergeLimit:
    def test_merge_limit_is_min_of_max_tokens_and_scaled_min_tokens(self) -> None:
        chunker = _make_chunker(min_tokens=100, max_merge_multiplier=2.0, max_tokens=1000)
        assert chunker.merge_limit == 200  # min(1000, 100*2.0)

    def test_merge_limit_capped_by_max_tokens(self) -> None:
        chunker = _make_chunker(min_tokens=100, max_merge_multiplier=20.0, max_tokens=1000)
        assert chunker.merge_limit == 1000  # min(1000, 100*20.0=2000) -> capped


class TestMergeLoop:
    def test_undersized_neighbors_merged_forward(self, monkeypatch) -> None:
        # min_tokens=5, max_merge_multiplier=10 -> merge_limit=50
        chunker = _make_chunker(min_tokens=5, max_merge_multiplier=10.0)
        c1 = _doc_chunk("one two", "#/texts/0")  # 2 tokens, undersized
        c2 = _doc_chunk("three four five", "#/texts/1")  # 3 tokens, undersized
        result = _run_chunk(chunker, [c1, c2], monkeypatch)

        assert len(result) == 1
        assert "one two" in result[0].text
        assert "three four five" in result[0].text

    def test_never_exceeds_max_tokens_via_merge_limit(self, monkeypatch) -> None:
        # merge_limit smaller than combined size stops the merge from growing further
        chunker = _make_chunker(min_tokens=3, max_merge_multiplier=1.5)  # merge_limit = 4
        c1 = _doc_chunk("a b", "#/texts/0")  # 2 tokens
        c2 = _doc_chunk("c d e f", "#/texts/1")  # 4 tokens -> combined would be 6 > merge_limit(4)
        result = _run_chunk(chunker, [c1, c2], monkeypatch)

        # merging c1+c2 would exceed merge_limit, so they stay separate
        assert len(result) == 2

    def test_single_oversized_chunk_untouched(self, monkeypatch) -> None:
        chunker = _make_chunker(min_tokens=3, max_merge_multiplier=2.0)
        big_text = " ".join(f"word{i}" for i in range(50))
        c1 = _doc_chunk(big_text, "#/texts/0")
        result = _run_chunk(chunker, [c1], monkeypatch)

        assert len(result) == 1
        assert result[0].text == big_text

    def test_no_undersized_chunks_passthrough(self, monkeypatch) -> None:
        chunker = _make_chunker(min_tokens=2, max_merge_multiplier=2.0)
        c1 = _doc_chunk("alpha beta gamma", "#/texts/0")  # 3 tokens, >= min_tokens
        c2 = _doc_chunk("delta epsilon zeta", "#/texts/1")  # 3 tokens, >= min_tokens
        result = _run_chunk(chunker, [c1, c2], monkeypatch)

        assert len(result) == 2
        assert result[0].text == "alpha beta gamma"
        assert result[1].text == "delta epsilon zeta"

    def test_trailing_undersized_group_merges_backward_into_previous(self, monkeypatch) -> None:
        # min_tokens=5, merge_limit generous enough to allow backward merge
        chunker = _make_chunker(min_tokens=5, max_merge_multiplier=10.0)  # merge_limit=50
        c1 = _doc_chunk("one two three four five six", "#/texts/0")  # 6 tokens, >= min_tokens
        c2 = _doc_chunk("seven", "#/texts/1")  # 1 token, undersized, no next chunk to merge forward
        result = _run_chunk(chunker, [c1, c2], monkeypatch)

        assert len(result) == 1
        assert "seven" in result[0].text


class _FakeDocItem:
    """Minimal stand-in exposing only the `.label` attribute `_infer_chunk_type` reads."""

    def __init__(self, label: DocItemLabel) -> None:
        self.label = label


class TestInferChunkType:
    def test_table_takes_precedence_over_picture(self) -> None:
        from src.services.ingestion.chunker import _infer_chunk_type

        table = _FakeDocItem(DocItemLabel.TABLE)
        picture = _FakeDocItem(DocItemLabel.PICTURE)
        assert _infer_chunk_type([table, picture]) == "table"
        assert _infer_chunk_type([picture]) == "picture"
        assert _infer_chunk_type([]) == "text"
