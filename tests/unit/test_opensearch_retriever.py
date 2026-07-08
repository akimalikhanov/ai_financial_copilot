"""Unit tests for OpenSearch keyword retrieval (mocked client)."""

from __future__ import annotations

from uuid import uuid4

import pytest

from src.services.retrieval import opensearch_retriever
from src.services.retrieval.opensearch_retriever import retrieve


class FakeClient:
    def __init__(self, hits: list[dict]) -> None:
        self._hits = hits
        self.calls: list[dict] = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return {"hits": {"hits": self._hits}}


def _hit(chunk_id=None, document_id=None, score: float | None = 1.5, **source_overrides):
    source = {
        "chunk_id": str(chunk_id or uuid4()),
        "document_id": str(document_id or uuid4()),
        "chunk_index": 0,
    }
    source.update(source_overrides)
    hit: dict = {"_source": source}
    if score is not None:
        hit["_score"] = score
    return hit


class TestRetrieveEdgeCases:
    @pytest.mark.asyncio
    async def test_top_k_zero_returns_empty_no_client_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = FakeClient([])
        monkeypatch.setattr(opensearch_retriever, "get_client", lambda: client)
        result = await retrieve("revenue", uuid4(), top_k=0)
        assert result == []
        assert client.calls == []

    @pytest.mark.asyncio
    async def test_empty_query_text_returns_empty_no_client_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = FakeClient([])
        monkeypatch.setattr(opensearch_retriever, "get_client", lambda: client)
        result = await retrieve("   ", uuid4(), top_k=10)
        assert result == []
        assert client.calls == []


class TestRetrieveHappyPath:
    @pytest.mark.asyncio
    async def test_correct_chunk_construction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cid, did = uuid4(), uuid4()
        hit = _hit(chunk_id=cid, document_id=did, score=3.2, page_start=1, page_end=2)
        client = FakeClient([hit])
        monkeypatch.setattr(opensearch_retriever, "get_client", lambda: client)

        result = await retrieve("revenue", uuid4(), top_k=5)

        assert len(result) == 1
        chunk = result[0]
        assert chunk.chunk_id == cid
        assert chunk.document_id == did
        assert chunk.source == "keyword"
        assert chunk.keyword_rank == 1
        assert chunk.keyword_score == 3.2
        assert chunk.score == 3.2


class TestMissingScore:
    @pytest.mark.asyncio
    async def test_missing_score_defaults_to_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hit = _hit(score=None)
        client = FakeClient([hit])
        monkeypatch.setattr(opensearch_retriever, "get_client", lambda: client)

        result = await retrieve("revenue", uuid4(), top_k=5)
        assert len(result) == 1
        assert result[0].score == 0.0
        assert result[0].keyword_score == 0.0


class TestMalformedHits:
    @pytest.mark.asyncio
    async def test_missing_chunk_id_filtered_out(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hit = {"_source": {"document_id": str(uuid4())}, "_score": 1.0}
        client = FakeClient([hit])
        monkeypatch.setattr(opensearch_retriever, "get_client", lambda: client)

        result = await retrieve("revenue", uuid4(), top_k=5)
        assert result == []

    @pytest.mark.asyncio
    async def test_missing_document_id_filtered_out(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hit = {"_source": {"chunk_id": str(uuid4())}, "_score": 1.0}
        client = FakeClient([hit])
        monkeypatch.setattr(opensearch_retriever, "get_client", lambda: client)

        result = await retrieve("revenue", uuid4(), top_k=5)
        assert result == []

    @pytest.mark.asyncio
    async def test_malformed_uuid_filtered_out(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hit = _hit()
        hit["_source"]["chunk_id"] = "not-a-uuid"
        client = FakeClient([hit])
        monkeypatch.setattr(opensearch_retriever, "get_client", lambda: client)

        result = await retrieve("revenue", uuid4(), top_k=5)
        assert result == []
