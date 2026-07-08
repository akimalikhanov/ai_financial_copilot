"""Unit tests for reranker (NoOpReranker, get_reranker, LocalCrossEncoderReranker)."""

from __future__ import annotations

from uuid import uuid4

import httpx
import pytest
import respx

from src.schemas.retrieval import RetrievedChunk
from src.services.retrieval import reranker as reranker_mod
from src.services.retrieval.reranker import LocalCrossEncoderReranker, NoOpReranker, get_reranker


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


class TestNoOpReranker:
    @pytest.mark.asyncio
    async def test_identity_passthrough(self) -> None:
        chunks = [_chunk(), _chunk()]
        result = await NoOpReranker().rerank("query", chunks, {})
        assert result == chunks


class TestGetReranker:
    def test_disabled_returns_noop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(reranker_mod, "get_reranker_enabled", lambda: False)
        assert isinstance(get_reranker(), NoOpReranker)

    def test_enabled_returns_local_cross_encoder(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(reranker_mod, "get_reranker_enabled", lambda: True)
        assert isinstance(get_reranker(), LocalCrossEncoderReranker)


BASE_URL = "http://test-reranker:8080"


class TestLocalCrossEncoderRerankerEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_chunks_no_http_call(self) -> None:
        r = LocalCrossEncoderReranker(base_url=BASE_URL)
        with respx.mock(assert_all_called=False) as mock:
            route = mock.post(f"{BASE_URL}/rerank")
            result = await r.rerank("q", [], {})
        assert result == []
        assert route.call_count == 0

    @pytest.mark.asyncio
    async def test_all_empty_texts_no_http_call(self) -> None:
        r = LocalCrossEncoderReranker(base_url=BASE_URL)
        chunks = [_chunk(), _chunk()]
        with respx.mock(assert_all_called=False) as mock:
            route = mock.post(f"{BASE_URL}/rerank")
            result = await r.rerank("q", chunks, {})
        assert result == chunks
        assert route.call_count == 0

    @pytest.mark.asyncio
    async def test_chunks_beyond_max_input_truncated_before_send(self) -> None:
        r = LocalCrossEncoderReranker(base_url=BASE_URL, max_input=2, rerank_top_k=10)
        chunks = [_chunk() for _ in range(5)]
        texts = {c.chunk_id: "text" for c in chunks}

        captured_body = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            captured_body.update(json.loads(request.content))
            return httpx.Response(
                200, json=[{"index": 0, "score": 0.9}, {"index": 1, "score": 0.1}]
            )

        with respx.mock:
            respx.post(f"{BASE_URL}/rerank").mock(side_effect=handler)
            await r.rerank("q", chunks, texts)

        assert len(captured_body["texts"]) == 2


class TestLocalCrossEncoderRerankerErrorHandling:
    @pytest.mark.asyncio
    async def test_http_error_falls_back_to_top_k(self, caplog: pytest.LogCaptureFixture) -> None:
        r = LocalCrossEncoderReranker(base_url=BASE_URL, rerank_top_k=2)
        chunks = [_chunk() for _ in range(5)]
        texts = {c.chunk_id: "text" for c in chunks}

        with respx.mock:
            respx.post(f"{BASE_URL}/rerank").mock(return_value=httpx.Response(500))
            with caplog.at_level("WARNING"):
                result = await r.rerank("q", chunks, texts)

        assert result == chunks[:2]
        assert "reranker_failed" in caplog.text

    @pytest.mark.asyncio
    async def test_timeout_falls_back_to_top_k(self, caplog: pytest.LogCaptureFixture) -> None:
        r = LocalCrossEncoderReranker(base_url=BASE_URL, rerank_top_k=2)
        chunks = [_chunk() for _ in range(3)]
        texts = {c.chunk_id: "text" for c in chunks}

        with respx.mock:
            respx.post(f"{BASE_URL}/rerank").mock(side_effect=httpx.TimeoutException("timeout"))
            with caplog.at_level("WARNING"):
                result = await r.rerank("q", chunks, texts)

        assert result == chunks[:2]
        assert "reranker_failed" in caplog.text

    @pytest.mark.asyncio
    async def test_malformed_response_not_a_list_falls_back(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        r = LocalCrossEncoderReranker(base_url=BASE_URL, rerank_top_k=2)
        chunks = [_chunk() for _ in range(3)]
        texts = {c.chunk_id: "text" for c in chunks}

        with respx.mock:
            respx.post(f"{BASE_URL}/rerank").mock(
                return_value=httpx.Response(200, json={"bad": "shape"})
            )
            with caplog.at_level("WARNING"):
                result = await r.rerank("q", chunks, texts)

        assert result == chunks[:2]
        assert "reranker_invalid_response" in caplog.text

    @pytest.mark.asyncio
    async def test_incomplete_scores_falls_back(self, caplog: pytest.LogCaptureFixture) -> None:
        r = LocalCrossEncoderReranker(base_url=BASE_URL, rerank_top_k=2)
        chunks = [_chunk() for _ in range(3)]
        texts = {c.chunk_id: "text" for c in chunks}

        with respx.mock:
            # only 1 score returned for 3 texts sent
            respx.post(f"{BASE_URL}/rerank").mock(
                return_value=httpx.Response(200, json=[{"index": 0, "score": 0.5}])
            )
            with caplog.at_level("WARNING"):
                result = await r.rerank("q", chunks, texts)

        assert result == chunks[:2]
        assert "reranker_incomplete_scores" in caplog.text


class TestLocalCrossEncoderRerankerHappyPath:
    @pytest.mark.asyncio
    async def test_reorders_by_descending_score_and_truncates(self) -> None:
        r = LocalCrossEncoderReranker(base_url=BASE_URL, rerank_top_k=2)
        chunks = [_chunk() for _ in range(3)]
        texts = {c.chunk_id: "text" for c in chunks}

        with respx.mock:
            respx.post(f"{BASE_URL}/rerank").mock(
                return_value=httpx.Response(
                    200,
                    json=[
                        {"index": 0, "score": 0.1},
                        {"index": 1, "score": 0.9},
                        {"index": 2, "score": 0.5},
                    ],
                )
            )
            result = await r.rerank("q", chunks, texts)

        assert len(result) == 2
        assert result[0].chunk_id == chunks[1].chunk_id
        assert result[0].score == 0.9
        assert result[1].chunk_id == chunks[2].chunk_id
        assert result[1].score == 0.5
