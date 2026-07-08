"""Unit tests for OpenSearch ingestion helpers (mocked client)."""

from __future__ import annotations

from uuid import uuid4

import pytest

from src.services.ingestion import opensearch_ingest


class FakeIndices:
    def __init__(self, exists: bool) -> None:
        self._exists = exists
        self.create_calls: list[dict] = []

    def exists(self, index: str) -> bool:  # noqa: ARG002
        return self._exists

    def create(self, **kwargs) -> None:
        self.create_calls.append(kwargs)


class FakeClient:
    def __init__(self, index_exists: bool = False) -> None:
        self.indices = FakeIndices(index_exists)


class TestEnsureIndex:
    def test_creates_index_when_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = FakeClient(index_exists=False)
        monkeypatch.setattr(opensearch_ingest, "get_client", lambda: client)
        opensearch_ingest.ensure_index("chunks")
        assert len(client.indices.create_calls) == 1

    def test_idempotent_skips_creation_when_exists(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = FakeClient(index_exists=True)
        monkeypatch.setattr(opensearch_ingest, "get_client", lambda: client)
        opensearch_ingest.ensure_index("chunks")
        assert client.indices.create_calls == []


class TestBulkIndex:
    def test_empty_input_no_client_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = []
        monkeypatch.setattr("opensearchpy.helpers.bulk", lambda *a, **k: calls.append((a, k)))
        opensearch_ingest.bulk_index("chunks", uuid4(), [], user_id=uuid4())
        assert calls == []

    def test_bulk_called_with_index_actions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = []
        monkeypatch.setattr("opensearchpy.helpers.bulk", lambda *a, **k: calls.append((a, k)))
        monkeypatch.setattr(opensearch_ingest, "get_client", lambda: FakeClient())

        chunk_id = uuid4()
        chunks = [
            {
                "chunk_id": chunk_id,
                "chunk_index": 0,
                "enriched_text": "some text",
            }
        ]
        opensearch_ingest.bulk_index("chunks", uuid4(), chunks, user_id=uuid4())

        assert len(calls) == 1
        actions = calls[0][0][1]
        assert actions[0]["_op_type"] == "index"
        assert actions[0]["_id"] == str(chunk_id)


class TestBulkDelete:
    def test_empty_input_no_client_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = []
        monkeypatch.setattr("opensearchpy.helpers.bulk", lambda *a, **k: calls.append((a, k)))
        opensearch_ingest.bulk_delete("chunks", [])
        assert calls == []

    def test_bulk_called_with_delete_actions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = []
        monkeypatch.setattr("opensearchpy.helpers.bulk", lambda *a, **k: calls.append((a, k)))
        monkeypatch.setattr(opensearch_ingest, "get_client", lambda: FakeClient())

        cid = uuid4()
        opensearch_ingest.bulk_delete("chunks", [cid])

        assert len(calls) == 1
        actions = calls[0][0][1]
        assert actions[0]["_op_type"] == "delete"
        assert actions[0]["_id"] == str(cid)
