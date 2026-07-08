"""Integration tests for Qdrant/OpenSearch ingest+retrieve against real services.

Requires Docker services `qdrant` and `opensearch` (see docker-compose.yml). These
prove real client behavior (filters, scoring, batching) that the mocked unit tests
in test_qdrant_{ingest,retriever}.py / test_opensearch_{ingest,retriever}.py cannot.
"""

from __future__ import annotations

import uuid

import pytest

from src.services.ingestion import opensearch_ingest, qdrant_ingest
from src.services.opensearch_client import reset_client as reset_opensearch_client
from src.services.qdrant_client import reset_client as reset_qdrant_client
from src.services.retrieval import opensearch_retriever, qdrant_retriever

pytestmark = pytest.mark.integration

_VECTOR_DIM = 8


def _vector(seed: int) -> list[float]:
    return [float((seed + i) % 7) / 7.0 for i in range(_VECTOR_DIM)]


@pytest.fixture
def qdrant_collection():
    reset_qdrant_client()
    name = f"test_collection_{uuid.uuid4().hex[:8]}"
    qdrant_ingest.ensure_collection(name, _VECTOR_DIM)
    yield name
    from src.services.qdrant_client import get_client

    get_client().delete_collection(collection_name=name)


@pytest.fixture
def opensearch_index():
    reset_opensearch_client()
    name = f"test_index_{uuid.uuid4().hex[:8]}"
    opensearch_ingest.ensure_index(name)
    yield name
    from src.services.opensearch_client import get_client

    get_client().indices.delete(index=name)


class TestQdrant:
    def test_ensure_collection_idempotent_and_queryable(self, qdrant_collection: str) -> None:
        qdrant_ingest.ensure_collection(qdrant_collection, _VECTOR_DIM)  # second call: no-op

        from src.services.qdrant_client import get_client

        assert get_client().collection_exists(collection_name=qdrant_collection)

    @pytest.mark.asyncio
    async def test_upsert_and_retrieve_round_trip(
        self, qdrant_collection: str, monkeypatch
    ) -> None:
        monkeypatch.setattr(qdrant_retriever, "_COLLECTION_NAME", qdrant_collection)
        user_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        chunk_ids = [uuid.uuid4() for _ in range(3)]

        items = [
            {
                "vector": _vector(i),
                "chunk_id": chunk_ids[i],
                "chunk_index": i,
                "chunk_type": "text",
                "page_start": i + 1,
                "page_end": i + 1,
                "heading_trail": ["Section A"],
            }
            for i in range(3)
        ]
        qdrant_ingest.upsert_chunks(qdrant_collection, doc_id, items, user_id=user_id)

        results = await qdrant_retriever.retrieve(_vector(0), user_id, top_k=10)

        assert len(results) == 3
        assert {r.chunk_id for r in results} == set(chunk_ids)
        assert all(r.source == "vector" for r in results)
        assert all(r.vector_rank is not None and r.vector_score is not None for r in results)
        # nearest to the query vector (seed=0) should rank first
        assert results[0].chunk_id == chunk_ids[0]

    @pytest.mark.asyncio
    async def test_delete_by_document_removes_points(
        self, qdrant_collection: str, monkeypatch
    ) -> None:
        monkeypatch.setattr(qdrant_retriever, "_COLLECTION_NAME", qdrant_collection)
        user_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        chunk_id = uuid.uuid4()

        qdrant_ingest.upsert_chunks(
            qdrant_collection,
            doc_id,
            [{"vector": _vector(0), "chunk_id": chunk_id, "chunk_index": 0}],
            user_id=user_id,
        )
        assert await qdrant_retriever.retrieve(_vector(0), user_id, top_k=10)

        qdrant_ingest.delete_by_document(qdrant_collection, doc_id)

        assert await qdrant_retriever.retrieve(_vector(0), user_id, top_k=10) == []

    @pytest.mark.asyncio
    async def test_delete_by_chunk_ids_removes_points(
        self, qdrant_collection: str, monkeypatch
    ) -> None:
        monkeypatch.setattr(qdrant_retriever, "_COLLECTION_NAME", qdrant_collection)
        user_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        chunk_ids = [uuid.uuid4(), uuid.uuid4()]

        qdrant_ingest.upsert_chunks(
            qdrant_collection,
            doc_id,
            [
                {"vector": _vector(i), "chunk_id": cid, "chunk_index": i}
                for i, cid in enumerate(chunk_ids)
            ],
            user_id=user_id,
        )

        qdrant_ingest.delete_by_chunk_ids(qdrant_collection, [chunk_ids[0]])

        results = await qdrant_retriever.retrieve(_vector(0), user_id, top_k=10)
        assert {r.chunk_id for r in results} == {chunk_ids[1]}

    def test_batch_upsert_splits_across_500_boundary(self, qdrant_collection: str) -> None:
        user_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        items = [
            {"vector": _vector(i), "chunk_id": uuid.uuid4(), "chunk_index": i} for i in range(600)
        ]

        qdrant_ingest.upsert_chunks(qdrant_collection, doc_id, items, user_id=user_id)

        from src.services.qdrant_client import get_client

        count = get_client().count(collection_name=qdrant_collection, exact=True).count
        assert count == 600


class TestOpenSearch:
    def test_ensure_index_idempotent_and_queryable(self, opensearch_index: str) -> None:
        opensearch_ingest.ensure_index(opensearch_index)  # second call: no-op

        from src.services.opensearch_client import get_client

        mapping = get_client().indices.get_mapping(index=opensearch_index)
        properties = mapping[opensearch_index]["mappings"]["properties"]
        assert "enriched_text" in properties

    @pytest.mark.asyncio
    async def test_bulk_index_and_retrieve_round_trip(
        self, opensearch_index: str, monkeypatch
    ) -> None:
        monkeypatch.setattr(opensearch_retriever, "_INDEX_NAME", opensearch_index)
        user_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        chunk_ids = [uuid.uuid4() for _ in range(2)]

        chunks = [
            {
                "chunk_id": chunk_ids[0],
                "chunk_index": 0,
                "enriched_text": "revenue grew due to strong quarterly performance",
                "page_start": 1,
                "page_end": 1,
            },
            {
                "chunk_id": chunk_ids[1],
                "chunk_index": 1,
                "enriched_text": "unrelated discussion about weather patterns",
                "page_start": 2,
                "page_end": 2,
            },
        ]
        opensearch_ingest.bulk_index(opensearch_index, doc_id, chunks, user_id=user_id)

        results = await opensearch_retriever.retrieve("quarterly revenue performance", user_id)

        assert len(results) >= 1
        assert results[0].chunk_id == chunk_ids[0]
        assert results[0].source == "keyword"
        assert results[0].keyword_score is not None and results[0].keyword_score > 0

    @pytest.mark.asyncio
    async def test_delete_by_document_removes_docs(
        self, opensearch_index: str, monkeypatch
    ) -> None:
        monkeypatch.setattr(opensearch_retriever, "_INDEX_NAME", opensearch_index)
        user_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        chunk_id = uuid.uuid4()

        opensearch_ingest.bulk_index(
            opensearch_index,
            doc_id,
            [{"chunk_id": chunk_id, "chunk_index": 0, "enriched_text": "quarterly revenue report"}],
            user_id=user_id,
        )
        assert await opensearch_retriever.retrieve("quarterly revenue", user_id)

        opensearch_ingest.delete_by_document(opensearch_index, doc_id)

        from src.services.opensearch_client import get_client

        get_client().indices.refresh(index=opensearch_index)
        assert await opensearch_retriever.retrieve("quarterly revenue", user_id) == []

    @pytest.mark.asyncio
    async def test_bulk_delete_removes_specific_chunks(
        self, opensearch_index: str, monkeypatch
    ) -> None:
        monkeypatch.setattr(opensearch_retriever, "_INDEX_NAME", opensearch_index)
        user_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        chunk_ids = [uuid.uuid4(), uuid.uuid4()]

        opensearch_ingest.bulk_index(
            opensearch_index,
            doc_id,
            [
                {"chunk_id": cid, "chunk_index": i, "enriched_text": "quarterly revenue report"}
                for i, cid in enumerate(chunk_ids)
            ],
            user_id=user_id,
        )

        opensearch_ingest.bulk_delete(opensearch_index, [chunk_ids[0]])

        results = await opensearch_retriever.retrieve("quarterly revenue", user_id, top_k=10)
        assert {r.chunk_id for r in results} == {chunk_ids[1]}
