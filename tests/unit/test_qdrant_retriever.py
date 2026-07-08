"""Unit tests for Qdrant vector retrieval (mocked client)."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from src.services.retrieval import qdrant_retriever
from src.services.retrieval.qdrant_retriever import retrieve


class FakeResponse:
    def __init__(self, points: list) -> None:
        self.points = points


class FakeClient:
    def __init__(self, points: list) -> None:
        self._points = points
        self.calls: list[dict] = []

    def query_points(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse(self._points)


def _point(chunk_id=None, document_id=None, score=0.9, **payload_overrides):
    payload = {
        "chunk_id": str(chunk_id or uuid4()),
        "document_id": str(document_id or uuid4()),
        "chunk_index": 0,
    }
    payload.update(payload_overrides)
    return SimpleNamespace(id=str(uuid4()), payload=payload, score=score)


class TestRetrieveEdgeCases:
    @pytest.mark.asyncio
    async def test_top_k_zero_returns_empty_no_client_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = FakeClient([])
        monkeypatch.setattr(qdrant_retriever, "get_client", lambda: client)
        result = await retrieve([0.1, 0.2], uuid4(), top_k=0)
        assert result == []
        assert client.calls == []

    @pytest.mark.asyncio
    async def test_empty_query_vector_returns_empty_no_client_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = FakeClient([])
        monkeypatch.setattr(qdrant_retriever, "get_client", lambda: client)
        result = await retrieve([], uuid4(), top_k=10)
        assert result == []
        assert client.calls == []


class TestRetrieveHappyPath:
    @pytest.mark.asyncio
    async def test_correct_chunk_construction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cid, did = uuid4(), uuid4()
        point = _point(chunk_id=cid, document_id=did, score=0.75, page_start=2, page_end=3)
        client = FakeClient([point])
        monkeypatch.setattr(qdrant_retriever, "get_client", lambda: client)

        result = await retrieve([0.1, 0.2], uuid4(), top_k=5)

        assert len(result) == 1
        chunk = result[0]
        assert chunk.chunk_id == cid
        assert chunk.document_id == did
        assert chunk.source == "vector"
        assert chunk.vector_rank == 1
        assert chunk.vector_score == 0.75
        assert chunk.score == 0.75
        assert chunk.page_start == 2
        assert chunk.page_end == 3


class TestMalformedPoints:
    @pytest.mark.asyncio
    async def test_missing_chunk_id_and_no_point_id_fallback_filtered(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # payload lacks chunk_id, but point.id provides a fallback — so to actually
        # trigger the None-filter we need document_id missing instead.
        point = SimpleNamespace(id=str(uuid4()), payload={"chunk_index": 0}, score=0.5)
        client = FakeClient([point])
        monkeypatch.setattr(qdrant_retriever, "get_client", lambda: client)

        result = await retrieve([0.1], uuid4(), top_k=5)
        assert result == []

    @pytest.mark.asyncio
    async def test_malformed_uuid_filtered_out(self, monkeypatch: pytest.MonkeyPatch) -> None:
        point = _point(chunk_id="not-a-uuid", document_id=uuid4())
        # overwrite chunk_id with genuinely invalid UUID string
        point.payload["chunk_id"] = "not-a-uuid"
        client = FakeClient([point])
        monkeypatch.setattr(qdrant_retriever, "get_client", lambda: client)

        result = await retrieve([0.1], uuid4(), top_k=5)
        assert result == []

    @pytest.mark.asyncio
    async def test_point_id_used_as_chunk_id_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        point_id = str(uuid4())
        did = uuid4()
        point = SimpleNamespace(
            id=point_id, payload={"document_id": str(did), "chunk_index": 0}, score=0.5
        )
        client = FakeClient([point])
        monkeypatch.setattr(qdrant_retriever, "get_client", lambda: client)

        result = await retrieve([0.1], uuid4(), top_k=5)
        assert len(result) == 1
        assert str(result[0].chunk_id) == point_id
