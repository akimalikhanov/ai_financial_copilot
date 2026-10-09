"""Ingestion backlog items P1-3..P1-5 and P2-1..P2-4 (docs/stages/ingestion-pipeline-findings.md).

P1-1 is covered in test_picture_enricher.py::TestStageTimeout, P1-3 in test_opensearch_ingest.py,
P2-2's embedder half in test_embedder.py.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pytest
from docling_core.transforms.chunker.doc_chunk import DocChunk, DocMeta
from docling_core.types.doc.document import DoclingDocument, TableData, TableItem
from docling_core.types.doc.labels import DocItemLabel

from src.services.ingestion import qdrant_ingest, s3_client, table_summarizer, tasks
from tests.unit.test_chunker import StubTokenizer, _doc_chunk, _make_chunker, _run_chunk

# -- P1-4 / P2-2: Qdrant ------------------------------------------------------------------


class _Qdrant:
    def __init__(self) -> None:
        self.upserts: list[dict[str, Any]] = []
        self.created: list[dict[str, Any]] = []

    def upsert(self, **kwargs: Any) -> None:
        self.upserts.append(kwargs)

    def collection_exists(self, collection_name: str) -> bool:  # noqa: ARG002
        return False

    def create_collection(self, **kwargs: Any) -> None:
        self.created.append(kwargs)

    def create_payload_index(self, *_a: Any) -> None:
        pass


class TestQdrantUpsert:
    def test_only_the_final_batch_waits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _Qdrant()
        monkeypatch.setattr(qdrant_ingest, "get_client", lambda: client)
        items = [{"vector": [0.1], "chunk_id": uuid4(), "chunk_index": i} for i in range(1200)]

        qdrant_ingest.upsert_chunks("docs", uuid4(), items, user_id=uuid4())

        assert [c["wait"] for c in client.upserts] == [False, False, True]
        assert [len(c["points"]) for c in client.upserts] == [500, 500, 200]

    def test_single_batch_waits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _Qdrant()
        monkeypatch.setattr(qdrant_ingest, "get_client", lambda: client)
        items = [{"vector": [0.1], "chunk_id": uuid4(), "chunk_index": 0}]

        qdrant_ingest.upsert_chunks("docs", uuid4(), items, user_id=uuid4())

        assert [c["wait"] for c in client.upserts] == [True]

    def test_float32_rows_are_sent_as_lists(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _Qdrant()
        monkeypatch.setattr(qdrant_ingest, "get_client", lambda: client)
        vectors = np.asarray([[0.5, 0.25], [1.0, 0.0]], dtype=np.float32)
        items = [
            {"vector": v, "chunk_id": uuid4(), "chunk_index": i} for i, v in enumerate(vectors)
        ]

        qdrant_ingest.upsert_chunks("docs", uuid4(), items, user_id=uuid4())

        assert [p.vector for p in client.upserts[0]["points"]] == [[0.5, 0.25], [1.0, 0.0]]

    def test_collection_is_created_single_shard(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The final-batch wait only fences earlier batches within one shard."""
        client = _Qdrant()
        monkeypatch.setattr(qdrant_ingest, "get_client", lambda: client)

        qdrant_ingest.ensure_collection("docs", 4)

        assert client.created[0]["shard_number"] == 1


# -- P1-5: table summarizer ----------------------------------------------------------------


class _Resp:
    def __init__(self, text: str) -> None:
        self.text = text


class _TableLLM:
    """Echoes each table's text back as its summary; fails any batch containing "boom"."""

    def __init__(self) -> None:
        self.live = 0
        self.peak = 0

    async def complete(self, messages: Any, **_params: Any) -> _Resp:
        self.live += 1
        self.peak = max(self.peak, self.live)
        try:
            await asyncio.sleep(0.05)
            body = messages[-1].content
            if "boom" in body:
                raise RuntimeError("provider exploded")
            parts = body.split("[TABLE ")[1:]
            summaries = [
                {"table_id": int(p.split("]", 1)[0]), "summary": "S:" + p.split("\n", 1)[1]}
                for p in parts
            ]
            return _Resp(json.dumps({"summaries": summaries}))
        finally:
            self.live -= 1


@pytest.fixture
def table_llm(monkeypatch: pytest.MonkeyPatch) -> _TableLLM:
    llm = _TableLLM()
    monkeypatch.setattr(table_summarizer, "_ensure_initialized", lambda: None)
    monkeypatch.setattr(table_summarizer, "_llm", llm)
    monkeypatch.setattr(table_summarizer, "_system_prompt", "sys")
    monkeypatch.setattr(table_summarizer, "_model_id", "m")
    monkeypatch.setattr(table_summarizer, "get_table_summarizer_batch_size", lambda: 1)
    monkeypatch.setattr(table_summarizer, "get_table_summarizer_concurrency", lambda: 3)
    return llm


class TestTableSummarizer:
    async def test_batches_overlap_under_the_cap(self, table_llm: _TableLLM) -> None:
        chunks = [{"chunk_type": "table", "enriched_text": f"t{i}"} for i in range(6)]

        await table_summarizer.summarize_table_chunks(chunks)

        assert table_llm.peak == 3

    async def test_summaries_land_on_their_own_chunks(self, table_llm: _TableLLM) -> None:  # noqa: ARG002
        chunks = [
            {"chunk_type": "table", "enriched_text": "a"},
            {"chunk_type": "text", "enriched_text": "prose"},
            {"chunk_type": "table", "enriched_text": "b"},
        ]

        await table_summarizer.summarize_table_chunks(chunks)

        assert [c.get("table_nl_summary") for c in chunks] == ["S:a", None, "S:b"]
        assert chunks[0]["table_nl_summary_model"] == "m"

    async def test_a_failed_batch_degrades_alone(self, table_llm: _TableLLM) -> None:  # noqa: ARG002
        chunks = [{"chunk_type": "table", "enriched_text": t} for t in ("a", "boom", "c")]

        await table_summarizer.summarize_table_chunks(chunks)

        assert [c["table_nl_summary"] for c in chunks] == ["S:a", None, "S:c"]
        assert chunks[1]["table_nl_summary_model"] is None


# -- P2-1: chunker -------------------------------------------------------------------------


def _split_segments(texts: list[str]) -> list[DocChunk]:
    """What docling emits for one oversized item: several chunks sharing its doc_items."""
    item = TableItem(self_ref="#/tables/0", label=DocItemLabel.TABLE, data=TableData())
    return [DocChunk(text=t, meta=DocMeta(doc_items=[item], headings=["H"])) for t in texts]


class TestChunker:
    def test_split_segments_render_their_own_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Keyed by doc_item refs alone, every segment rendered the last one's text — 98 of
        1765 chunks on the Poste Italiane filing were indexed under another segment's text."""
        chunker = _make_chunker(min_tokens=1, max_merge_multiplier=1.0)
        first = " ".join(["alpha"] * 5)
        second = " ".join(["omega"] * 5)

        out = _run_chunk(chunker, _split_segments([first, second]), monkeypatch)

        assert [chunker.contextualize(c) for c in out] == [
            f"[SECTION] H\n{first}",
            f"[SECTION] H\n{second}",
        ]

    def test_a_merge_candidate_is_not_tokenized(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Only accepted merges are rendered and counted; candidates are costed from their
        members' cached counts."""
        chunker = _make_chunker(min_tokens=10, max_merge_multiplier=2.0)
        counted: list[str] = []
        real = StubTokenizer.count_tokens

        def _spy(self: StubTokenizer, text: str) -> int:
            counted.append(text)
            return real(self, text)

        monkeypatch.setattr(StubTokenizer, "count_tokens", _spy)
        pieces = [_doc_chunk(f"w{i} x", f"#/texts/{i}") for i in range(6)]

        out = _run_chunk(chunker, pieces, monkeypatch)

        # merge_limit is 20 and each piece renders as 4 words: five merge, the sixth cannot.
        assert len(out) == 2
        # The delimiter, 6 members, and the one accepted merge; the old loop also rendered
        # every growing candidate and re-counted members on each pass.
        assert len(counted) == 8

    def test_merge_still_respects_the_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        chunker = _make_chunker(min_tokens=5, max_merge_multiplier=2.0)  # merge_limit = 10
        pieces = [_doc_chunk("a b c", f"#/texts/{i}", heading="") for i in range(4)]

        out = _run_chunk(chunker, pieces, monkeypatch)

        assert [len(c.text.split()) for c in out] == [9, 3]


# -- P2-3: purge ---------------------------------------------------------------------------


def test_purge_is_skipped_on_the_first_attempt() -> None:
    source = inspect.getsource(tasks._run_pipeline)
    stage = source.split('_log_stage("purge_stale_chunks")')[1].split("_log_stage(")[0]
    assert "if attempt > 1:" in stage
    assert stage.index("if attempt > 1:") < stage.index("delete_by_document")


# -- P2-4: resume --------------------------------------------------------------------------


class TestResume:
    async def test_loads_the_persisted_document(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        doc = DoclingDocument(name="d")
        doc.add_text(label=DocItemLabel.TEXT, text="hello")
        path = tmp_path / "docling.json"
        doc.save_as_json(path)
        seen: list[tuple[str, str | None]] = []

        async def _download(key: str, *, bucket: str | None = None) -> Path:
            seen.append((key, bucket))
            return path

        monkeypatch.setattr(s3_client, "download_file", _download)

        loaded = await tasks._load_persisted_document("k/docling.json", "doc-1")

        assert loaded is not None and loaded.texts[0].text == "hello"
        assert seen == [("k/docling.json", tasks.get_s3_docling_bucket())]
        assert not path.exists(), "the downloaded copy must be removed"

    async def test_missing_artifact_falls_back_to_a_parse(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _download(_key: str, *, bucket: str | None = None) -> Path:  # noqa: ARG001
            raise RuntimeError("NoSuchKey")

        monkeypatch.setattr(s3_client, "download_file", _download)

        assert await tasks._load_persisted_document("k", "doc-1") is None

    async def test_corrupt_artifact_falls_back_to_a_parse(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "docling.json"
        path.write_text("{truncated")

        async def _download(_key: str, *, bucket: str | None = None) -> Path:  # noqa: ARG001
            return path

        monkeypatch.setattr(s3_client, "download_file", _download)

        assert await tasks._load_persisted_document("k", "doc-1") is None
        assert not path.exists()

    def test_only_a_retry_with_a_recorded_parse_resumes(self) -> None:
        """parse_status is written alongside the docling.json upload, so it marks a parse that
        was persisted; attempt 1 never has one."""
        source = inspect.getsource(tasks._run_pipeline)
        branch = source.split("# -- resume:")[1].split('_log_stage("download_pdf")')[0]
        assert "attempt > 1 and prior_parse_status is not None" in branch
        assert "_load_persisted_document(" in branch
        # Resume replaces the whole parse path, including the download.
        assert source.index("# -- resume:") < source.index('_log_stage("download_pdf")')
        # And the chunker reads whichever document was produced.
        assert "chunker.chunk_document, document, document_id" in source
