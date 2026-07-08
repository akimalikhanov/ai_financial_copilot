"""Unit tests for Qdrant ingestion helpers (mocked client)."""

from __future__ import annotations

from uuid import uuid4

import pytest

from src.services.ingestion import qdrant_ingest


class FakeClient:
    def __init__(self, collection_exists: bool = False, raise_on_index: bool = False) -> None:
        self._collection_exists = collection_exists
        self._raise_on_index = raise_on_index
        self.created_collections: list[dict] = []
        self.created_indexes: list[tuple] = []
        self.upsert_calls: list[dict] = []

    def collection_exists(self, collection_name: str) -> bool:  # noqa: ARG002
        return self._collection_exists

    def create_collection(self, **kwargs) -> None:
        self.created_collections.append(kwargs)

    def create_payload_index(self, name, field, schema_type) -> None:
        if self._raise_on_index:
            raise RuntimeError("index already exists")
        self.created_indexes.append((name, field, schema_type))

    def upsert(self, **kwargs) -> None:
        self.upsert_calls.append(kwargs)


class TestEnsureCollection:
    def test_creates_collection_when_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = FakeClient(collection_exists=False)
        monkeypatch.setattr(qdrant_ingest, "get_client", lambda: client)
        qdrant_ingest.ensure_collection("docs", 768)
        assert len(client.created_collections) == 1
        assert len(client.created_indexes) == 2

    def test_idempotent_skips_creation_when_exists(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = FakeClient(collection_exists=True)
        monkeypatch.setattr(qdrant_ingest, "get_client", lambda: client)
        qdrant_ingest.ensure_collection("docs", 768)
        assert client.created_collections == []

    def test_payload_index_exception_suppressed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = FakeClient(collection_exists=True, raise_on_index=True)
        monkeypatch.setattr(qdrant_ingest, "get_client", lambda: client)
        qdrant_ingest.ensure_collection("docs", 768)  # must not raise
        assert client.created_indexes == []


class TestUpsertChunks:
    def test_empty_input_no_client_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = FakeClient()
        monkeypatch.setattr(qdrant_ingest, "get_client", lambda: client)
        qdrant_ingest.upsert_chunks("docs", uuid4(), [], user_id=uuid4())
        assert client.upsert_calls == []

    def test_batch_of_500_splitting(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = FakeClient()
        monkeypatch.setattr(qdrant_ingest, "get_client", lambda: client)

        items = [{"vector": [0.1, 0.2], "chunk_id": uuid4(), "chunk_index": i} for i in range(600)]
        qdrant_ingest.upsert_chunks("docs", uuid4(), items, user_id=uuid4())

        assert len(client.upsert_calls) == 2
        assert len(client.upsert_calls[0]["points"]) == 500
        assert len(client.upsert_calls[1]["points"]) == 100

    def test_single_batch_under_500(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = FakeClient()
        monkeypatch.setattr(qdrant_ingest, "get_client", lambda: client)

        items = [{"vector": [0.1], "chunk_id": uuid4(), "chunk_index": 0}]
        qdrant_ingest.upsert_chunks("docs", uuid4(), items, user_id=uuid4())

        assert len(client.upsert_calls) == 1
        assert len(client.upsert_calls[0]["points"]) == 1
