"""Post-parse memory fixes in the ingestion pipeline (findings doc §3 P0-1..P0-4).

Crop upload is bounded and runs before export, export writes to disk with no embedded crops,
the chunk backup streams to a file, and the worker shares one S3 client.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
from typing import Any

import pytest
from docling_core.types.doc.document import (
    DoclingDocument,
    ImageRef,
    PictureClassificationMetaField,
    PictureClassificationPrediction,
    PictureItem,
    PictureMeta,
)
from PIL import Image

from src.services.ingestion import picture_enricher, s3_client, tasks


def _doc(n: int) -> DoclingDocument:
    doc = DoclingDocument(name="test")
    for i in range(n):
        doc.pictures.append(
            PictureItem(
                self_ref=f"#/pictures/{i}",
                image=ImageRef.from_pil(Image.new("RGB", (8, 8), "red"), dpi=72),
                meta=PictureMeta(
                    classification=PictureClassificationMetaField(
                        predictions=[
                            PictureClassificationPrediction(class_name="bar_chart", confidence=0.9)
                        ]
                    )
                ),
                prov=[],
            )
        )
    return doc


class TestExportArtifacts:
    def test_docling_json_carries_no_base64_crops(self, tmp_path: Path) -> None:
        """PLACEHOLDER only strips images from Markdown; save_as_json embeds them regardless."""
        doc = _doc(2)
        json_path, md_path = tasks._export_artifacts(doc, tmp_path)

        body = json_path.read_text()
        assert "data:image/png;base64" not in body
        assert "data:image/png;base64" not in md_path.read_text()
        assert len(json.loads(body)["pictures"]) == 2
        assert all(pic.image is None for pic in doc.pictures)

    def test_docling_json_is_not_indented(self, tmp_path: Path) -> None:
        json_path, _ = tasks._export_artifacts(_doc(1), tmp_path)
        assert "\n" not in json_path.read_text()


class TestUploadPictureCrops:
    async def test_uploads_are_bounded_and_carry_classification(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        live = peak = 0
        uploaded: list[dict[str, Any]] = []

        async def _fake_upload(_document_id: str, self_ref: str, data: bytes, **kw: Any) -> str:
            nonlocal live, peak
            live += 1
            peak = max(peak, live)
            try:
                await asyncio.sleep(0.01)
                uploaded.append({"self_ref": self_ref, "data": data, **kw})
            finally:
                live -= 1
            return self_ref

        monkeypatch.setattr(s3_client, "upload_picture_crop", _fake_upload)
        monkeypatch.setattr(tasks, "_CROP_UPLOAD_CONCURRENCY", 3)

        doc = _doc(10)
        assert await tasks._upload_picture_crops(doc, "doc-1") == 10

        assert peak <= 3, f"crop uploads unbounded: peak {peak}"
        assert {u["self_ref"] for u in uploaded} == {f"#/pictures/{i}" for i in range(10)}
        assert all(u["data"].startswith(b"\x89PNG") for u in uploaded)
        assert all(u["label"] == "bar_chart" for u in uploaded)
        # Upload does not strip the crops; export does.
        assert all(pic.image is not None for pic in doc.pictures)

    async def test_one_failed_upload_does_not_stop_the_rest(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _flaky(_document_id: str, self_ref: str, _data: bytes, **_kw: Any) -> str:
            if self_ref == "#/pictures/1":
                raise RuntimeError("garage timeout")
            return self_ref

        monkeypatch.setattr(s3_client, "upload_picture_crop", _flaky)
        doc = _doc(3)
        doc.pictures[2].image = None  # no crop at all: skipped, not counted

        assert await tasks._upload_picture_crops(doc, "doc-1") == 1

    def test_pipeline_uploads_crops_before_export(self) -> None:
        """Export clears pic.image, so crops must already be uploaded; and the two must not be
        gathered, or their peaks add."""
        source = inspect.getsource(tasks._run_pipeline)
        upload_at = source.index('_log_stage("upload_picture_crops")')
        export_at = source.index('_log_stage("export_docling_artifacts")')
        assert upload_at < export_at
        export_stage = source[export_at : source.index('_log_stage("save_metadata_and_upload')]
        assert "gather" not in export_stage


class TestChunksJsonl:
    def test_rows_stream_to_file(self, tmp_path: Path) -> None:
        class _Db:
            def __init__(self, i: int) -> None:
                self.id = f"id-{i}"

        chunks = [
            {"chunk_index": i, "raw_text": f"r{i}", "enriched_text": f"e{i}"} for i in range(3)
        ]
        path = tmp_path / "chunks.jsonl"
        tasks._write_chunks_jsonl(path, chunks, [_Db(i) for i in range(3)])

        rows = [json.loads(line) for line in path.read_text().split("\n")]
        assert [r["chunk_id"] for r in rows] == ["id-0", "id-1", "id-2"]
        assert rows[1]["enriched_text"] == "e1"


class TestSharedS3Client:
    async def test_one_client_per_loop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        opened: list[object] = []
        closed: list[object] = []

        class _Client:
            async def __aenter__(self) -> _Client:
                opened.append(self)
                await asyncio.sleep(0)
                return self

            async def __aexit__(self, *_exc: Any) -> None:
                closed.append(self)

        monkeypatch.setattr(s3_client, "_new_client_cm", lambda _config: _Client())
        s3_client.reset_client()

        clients = await asyncio.gather(*(s3_client._get_client() for _ in range(20)))

        assert len(opened) == 1, "concurrent callers each opened a client"
        assert all(c is clients[0] for c in clients)

        await s3_client.close_client()
        assert closed == opened
        s3_client.reset_client()


class TestEnricherEncodesLazily:
    async def test_plan_does_not_encode_crops(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Encoding every crop up front held all of them for the whole stage."""

        def _boom(_pic: Any) -> bytes:
            raise AssertionError("_plan encoded a crop")

        monkeypatch.setattr(picture_enricher, "_crop_bytes", _boom)
        charts, captions = picture_enricher._plan(_doc(3), 0.0)
        assert len(charts) == 3 and captions == []
