"""Unit tests for RRF fusion (pure function, no mocking)."""

from __future__ import annotations

from uuid import uuid4

import pytest

from src.schemas.retrieval import RetrievedChunk
from src.services.retrieval.hybrid_retriever import fuse_rrf


def _chunk(**overrides) -> RetrievedChunk:
    defaults = {
        "chunk_id": uuid4(),
        "document_id": uuid4(),
        "score": 1.0,
        "chunk_index": 0,
        "page_start": 1,
        "page_end": 1,
        "heading_trail": [],
        "source": "vector",
    }
    defaults.update(overrides)
    return RetrievedChunk(**defaults)


class TestFuseRrfEdgeCases:
    def test_final_top_k_zero_returns_empty(self) -> None:
        assert fuse_rrf([_chunk()], [], final_top_k=0) == []

    def test_final_top_k_negative_returns_empty(self) -> None:
        assert fuse_rrf([_chunk()], [], final_top_k=-5) == []

    def test_negative_k_raises(self) -> None:
        with pytest.raises(ValueError, match="k must be non-negative"):
            fuse_rrf([_chunk()], [], k=-1)

    def test_negative_vector_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="RRF weights must be non-negative"):
            fuse_rrf([_chunk()], [], vector_weight=-0.1, keyword_weight=0.5)

    def test_negative_keyword_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="RRF weights must be non-negative"):
            fuse_rrf([_chunk()], [], vector_weight=0.5, keyword_weight=-0.1)

    def test_zero_vector_weight_contributes_nothing(self) -> None:
        v = [_chunk()]
        k = [_chunk()]
        result = fuse_rrf(v, k, vector_weight=0.0, keyword_weight=1.0, k=60, final_top_k=10)
        assert len(result) == 1
        assert result[0].chunk_id == k[0].chunk_id

    def test_empty_vector_results_contributes_nothing(self) -> None:
        k = [_chunk()]
        result = fuse_rrf([], k, vector_weight=0.6, keyword_weight=0.4, k=60, final_top_k=10)
        assert len(result) == 1
        assert result[0].chunk_id == k[0].chunk_id

    def test_both_empty_returns_empty(self) -> None:
        assert fuse_rrf([], [], final_top_k=10) == []


class TestFuseRrfFusion:
    def test_chunk_in_both_lists_sums_contributions(self) -> None:
        cid = uuid4()
        v_chunk = _chunk(chunk_id=cid, source="vector")
        k_chunk = _chunk(chunk_id=cid, source="keyword")
        other = _chunk(source="keyword")

        result = fuse_rrf(
            [v_chunk], [k_chunk, other], vector_weight=0.6, keyword_weight=0.4, k=60, final_top_k=10
        )

        fused = next(c for c in result if c.chunk_id == cid)
        expected_score = 0.6 / (60 + 1) + 0.4 / (60 + 1)
        assert fused.score == pytest.approx(expected_score)
        assert fused.vector_rank == 1
        assert fused.vector_score == v_chunk.score
        assert fused.keyword_rank == 1
        assert fused.keyword_score == k_chunk.score
        assert fused.source == "hybrid"

    def test_duplicate_chunk_id_within_single_list_deduped(self) -> None:
        cid = uuid4()
        dup1 = _chunk(chunk_id=cid, score=0.9)
        dup2 = _chunk(chunk_id=cid, score=0.1)
        result = fuse_rrf(
            [dup1, dup2], [], vector_weight=1.0, keyword_weight=0.0, k=60, final_top_k=10
        )
        assert len(result) == 1
        # first occurrence's rank (1) is used, not the second's
        expected_score = 1.0 / (60 + 1)
        assert result[0].score == pytest.approx(expected_score)

    def test_truncation_and_descending_order(self) -> None:
        chunks = [_chunk() for _ in range(5)]
        result = fuse_rrf(chunks, [], vector_weight=1.0, keyword_weight=0.0, k=60, final_top_k=3)
        assert len(result) == 3
        scores = [c.score for c in result]
        assert scores == sorted(scores, reverse=True)

    def test_only_in_one_list_has_none_for_other_meta(self) -> None:
        v_only = _chunk()
        result = fuse_rrf([v_only], [], vector_weight=0.6, keyword_weight=0.4, k=60, final_top_k=10)
        assert result[0].keyword_rank is None
        assert result[0].keyword_score is None
