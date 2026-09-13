"""Partial-degradation behaviour of the chat RAG pipeline.

Embedding sits *in front of* both retrieval backends rather than beside them — it
produces an argument to the fan-out — so an unguarded failure there took keyword search
down with it while OpenSearch was healthy. These tests pin the fail-open and the
per-capability flags that let callers tell a half-broken search from a healthy one.
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from src.schemas.query_transform import TransformedQuery
from src.schemas.retrieval import RetrievedChunk
from src.services.retrieval import chat_rag
from src.services.retrieval.reranker import RerankOutcome


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        score=0.5,
        chunk_index=0,
        page_start=1,
        page_end=1,
        heading_trail=[],
        source="keyword",
    )


class _StubReranker:
    """Passes chunks through with cross-encoder-scale scores, as a healthy reranker would."""

    def __init__(self, outcome: RerankOutcome | None = None) -> None:
        self._outcome = outcome

    async def rerank(self, query, chunks, texts):  # noqa: ANN001, ARG002
        return self._outcome or RerankOutcome(chunks=list(chunks), scored=True)

    async def aclose(self) -> None:
        pass


@pytest.fixture
def _patched(monkeypatch: pytest.MonkeyPatch):
    """Neutralise everything after fusion so the tests speak only about degradation."""
    calls: dict[str, int] = {"qdrant": 0, "opensearch": 0}
    kw_chunks = [_chunk()]

    async def _qdrant(*args, **kwargs):  # noqa: ANN002, ANN003, ARG001
        calls["qdrant"] += 1
        return []

    async def _opensearch(*args, **kwargs):  # noqa: ANN002, ANN003, ARG001
        calls["opensearch"] += 1
        return list(kw_chunks)

    async def _payloads(*args, **kwargs):  # noqa: ANN002, ANN003, ARG001
        return {}

    monkeypatch.setattr(chat_rag, "qdrant_retrieve", _qdrant)
    monkeypatch.setattr(chat_rag, "opensearch_retrieve", _opensearch)
    monkeypatch.setattr(chat_rag, "get_chunk_prompt_payloads", _payloads)
    monkeypatch.setattr(
        chat_rag,
        "assemble_rag_context",
        lambda chunks, payloads: (  # noqa: ARG005
            SimpleNamespace(formatted_context="", items=tuple(chunks), chunk_count=len(chunks)),
            SimpleNamespace(dropped=[], flagged=[]),
        ),
    )
    return calls, kw_chunks


async def _run(**kwargs):
    return await chat_rag.run_chat_rag_pipeline(
        None,  # type: ignore[arg-type]  — session only reaches the patched hydrator
        transformed=TransformedQuery(semantic_query="q", keyword_query="q"),
        user_id=uuid4(),
        doc_ids=None,
        reranker=_StubReranker(),
        **kwargs,
    )


class TestEmbedFailOpen:
    @pytest.mark.asyncio
    async def test_dead_embedder_keeps_keyword_search_alive(
        self, monkeypatch: pytest.MonkeyPatch, _patched
    ) -> None:
        calls, kw_chunks = _patched

        def _boom(_text: str):
            raise RuntimeError("embedder unreachable")

        monkeypatch.setattr(chat_rag, "embed_query", _boom)

        _ctx, trace, chunks = await _run()

        # The whole point: OpenSearch is healthy, so the search still returns results.
        assert calls["opensearch"] == 1
        assert [c.chunk_id for c in chunks] == [c.chunk_id for c in kw_chunks]
        # Qdrant is skipped rather than queried with a bogus vector.
        assert calls["qdrant"] == 0
        # Degraded, but emphatically not a total outage — that distinction drives the
        # "search unavailable" gap text, which would be a false alarm here.
        assert trace.all_backends_failed is False
        assert trace.embed_ok is False
        assert trace.vector_ok is False
        assert trace.keyword_ok is True

    @pytest.mark.asyncio
    async def test_embed_and_keyword_both_dead_is_a_total_outage(
        self, monkeypatch: pytest.MonkeyPatch, _patched
    ) -> None:
        _calls, _kw = _patched

        def _boom(_text: str):
            raise RuntimeError("embedder unreachable")

        async def _os_boom(*args, **kwargs):  # noqa: ANN002, ANN003, ARG001
            raise RuntimeError("opensearch unreachable")

        monkeypatch.setattr(chat_rag, "embed_query", _boom)
        monkeypatch.setattr(chat_rag, "opensearch_retrieve", _os_boom)

        _ctx, trace, chunks = await _run()

        assert chunks == []
        assert trace.all_backends_failed is True
        assert trace.embed_ok is False

    @pytest.mark.asyncio
    async def test_healthy_run_reports_no_degradation(
        self, monkeypatch: pytest.MonkeyPatch, _patched
    ) -> None:
        _calls, _kw = _patched
        monkeypatch.setattr(chat_rag, "embed_query", lambda _t: [0.1, 0.2])

        _ctx, trace, _chunks = await _run()

        assert (trace.embed_ok, trace.vector_ok, trace.keyword_ok) == (True, True, True)
        assert trace.rerank_ok is True
        assert trace.scores_are_rerank is True


class TestRerankDegradation:
    @pytest.mark.asyncio
    async def test_fallen_open_reranker_marks_scores_unusable(
        self, monkeypatch: pytest.MonkeyPatch, _patched
    ) -> None:
        """Scores are then RRF-scale, which confidence thresholds must not be applied to."""
        _calls, _kw = _patched
        monkeypatch.setattr(chat_rag, "embed_query", lambda _t: [0.1, 0.2])

        _ctx, trace, _chunks = await chat_rag.run_chat_rag_pipeline(
            None,  # type: ignore[arg-type]
            transformed=TransformedQuery(semantic_query="q", keyword_query="q"),
            user_id=uuid4(),
            doc_ids=None,
            reranker=_StubReranker(RerankOutcome(chunks=[_chunk()], scored=False, degraded=True)),
        )

        assert trace.rerank_ok is False
        assert trace.scores_are_rerank is False
        # Retrieval itself was fine — only the ranking was lost.
        assert trace.all_backends_failed is False
        assert trace.keyword_ok is True
