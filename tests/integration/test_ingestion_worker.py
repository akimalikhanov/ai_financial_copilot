"""
Integration test: ingestion worker pipeline with mocked external services.

Requires: PostgreSQL running (docker-compose up -d postgres pgbouncer).

Tests the full ingestion pipeline orchestration with:
- Real PostgreSQL for document/chunk records
- Mocked: S3, Docling parser, chunker, embedder, Qdrant, OpenSearch
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from src.models.document import Document
from src.models.user import User
from src.repository.chunk_repository import ChunkRepository
from src.repository.document_repository import DocumentRepository
from src.utils.config import get_db_url


class MockParseResult:
    """Minimal mock for docling parse result."""

    def __init__(self) -> None:
        self.page_count = 3
        self.extracted_title = "Test Financial Report 10-K"
        self.parse_status = "success"
        self.metadata = {"author": "Test Corp", "year": "2025"}
        self.document = MagicMock()


def _create_mock_chunks() -> list[dict]:
    """Sample chunks that would come from chunker."""
    return [
        {
            "chunk_index": 0,
            "raw_text": "Revenue increased 15% year-over-year.",
            "enriched_text": "Financial Highlights > Revenue: Revenue increased 15% year-over-year.",
            "heading_trail": ["Financial Highlights", "Revenue"],
            "chunk_type": "text",
            "page_start": 1,
            "page_end": 1,
            "token_count": 42,
            "provenance": {"source": "docling"},
            "metadata": {},
        },
        {
            "chunk_index": 1,
            "raw_text": "Operating expenses decreased by 8%.",
            "enriched_text": "Financial Highlights > Expenses: Operating expenses decreased by 8%.",
            "heading_trail": ["Financial Highlights", "Expenses"],
            "chunk_type": "text",
            "page_start": 2,
            "page_end": 2,
            "token_count": 38,
            "provenance": {"source": "docling"},
            "metadata": {},
        },
    ]


@pytest.fixture
async def ingestion_session() -> AsyncGenerator[AsyncSession, None]:
    """Create a fresh session for ingestion tests."""
    engine = create_async_engine(get_db_url(), poolclass=NullPool)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session
    await engine.dispose()


@pytest.fixture
async def test_user(ingestion_session: AsyncSession) -> User:
    """Create a test user for document ownership."""
    user = User(
        email=f"ingestion-test-{uuid4().hex[:8]}@test.com",
        password_hash="fake-hash-not-used",
    )
    ingestion_session.add(user)
    await ingestion_session.flush()
    return user


@pytest.fixture
async def pending_document(ingestion_session: AsyncSession, test_user: User) -> Document:
    """Create a pending document record for ingestion test."""
    repo = DocumentRepository(ingestion_session)
    doc = await repo.create(
        user_id=test_user.id,
        original_filename="test_annual_report.pdf",
        storage_key=f"uploads/{uuid4()}/test_annual_report.pdf",
        file_size_bytes=1024 * 500,
    )
    await ingestion_session.commit()
    return doc


@pytest.fixture
def _mock_ingestion_services():
    """Patch all external ingestion services with mocks."""
    mock_chunks = _create_mock_chunks()
    fake_embeddings = [[0.1] * 384, [0.2] * 384]
    fake_artifacts = (b'{"mock": "json"}', b"# Mock Markdown")

    with (
        patch("src.services.ingestion.s3_client.download_file") as mock_s3_download,
        patch("src.services.ingestion.s3_client.upload_bytes") as mock_s3_upload,
        patch("src.services.ingestion.docling_parser.parse") as mock_parse,
        patch("src.services.ingestion.tasks._export_artifacts") as mock_export,
        patch("src.services.ingestion.chunker.chunk_document") as mock_chunk,
        patch("src.services.ingestion.embedder.embed_chunks") as mock_embed,
        patch("src.services.ingestion.qdrant_ingest.ensure_collection") as mock_qdrant_ensure,
        patch("src.services.ingestion.qdrant_ingest.delete_by_chunk_ids") as mock_qdrant_delete,
        patch("src.services.ingestion.qdrant_ingest.upsert_chunks") as mock_qdrant_upsert,
        patch("src.services.ingestion.opensearch_ingest.ensure_index") as mock_os_ensure,
        patch("src.services.ingestion.opensearch_ingest.bulk_delete") as mock_os_delete,
        patch("src.services.ingestion.opensearch_ingest.bulk_index") as mock_os_bulk,
    ):
        mock_s3_download.return_value = Path("/tmp/fake_doc.pdf")
        mock_s3_upload.return_value = None
        mock_parse.return_value = MockParseResult()
        mock_export.return_value = fake_artifacts
        mock_chunk.return_value = mock_chunks
        mock_embed.return_value = fake_embeddings
        mock_qdrant_ensure.return_value = None
        mock_qdrant_delete.return_value = None
        mock_qdrant_upsert.return_value = None
        mock_os_ensure.return_value = None
        mock_os_delete.return_value = None
        mock_os_bulk.return_value = None

        yield {
            "s3_download": mock_s3_download,
            "s3_upload": mock_s3_upload,
            "parse": mock_parse,
            "export": mock_export,
            "chunk": mock_chunk,
            "embed": mock_embed,
            "chunks": mock_chunks,
        }


@pytest.fixture
def _patch_session_factory(ingestion_session: AsyncSession):  # noqa: ARG001
    """Patch the ingestion task's session factory to use our test session."""
    engine = create_async_engine(get_db_url(), poolclass=NullPool)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    with patch("src.services.ingestion.tasks._session_factory", factory):
        yield factory


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ingest_document_happy_path(
    ingestion_session: AsyncSession,
    pending_document: Document,
    _mock_ingestion_services: dict,
    _patch_session_factory,
) -> None:
    """
    Full pipeline: pending → processing → ready.

    Verifies:
    - Document status transitions to 'ready'
    - Metadata is populated (page_count, extracted_title)
    - Chunks are persisted to Postgres
    - Timing info is recorded
    """
    doc_id = str(pending_document.id)

    from src.services.ingestion.tasks import _run_pipeline

    await _run_pipeline(doc_id)

    await ingestion_session.refresh(pending_document)

    assert pending_document.status == "ready"
    assert pending_document.page_count == 3
    assert pending_document.extracted_title == "Test Financial Report 10-K"
    assert pending_document.ingest_time_seconds is not None
    assert "stages" in pending_document.ingest_time_seconds
    assert "total_time" in pending_document.ingest_time_seconds

    chunk_repo = ChunkRepository(ingestion_session)
    chunks = await chunk_repo.list_by_document(pending_document.id)
    assert len(chunks) == 2
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ingest_document_not_found(_patch_session_factory) -> None:
    """Pipeline raises LookupError when document doesn't exist."""
    from src.services.ingestion.tasks import _run_pipeline

    fake_id = str(uuid4())
    with pytest.raises(LookupError, match="not found"):
        await _run_pipeline(fake_id)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ingest_document_s3_failure_sets_failed_status(
    ingestion_session: AsyncSession,
    pending_document: Document,
    _patch_session_factory,
) -> None:
    """Document status → 'failed' with error message when S3 download fails."""
    doc_id = str(pending_document.id)

    with patch(
        "src.services.ingestion.s3_client.download_file",
        side_effect=Exception("S3 connection timeout"),
    ):
        from src.services.ingestion.tasks import _run_pipeline

        with pytest.raises(Exception, match="S3 connection timeout"):
            await _run_pipeline(doc_id)

    await ingestion_session.refresh(pending_document)

    assert pending_document.status == "failed"
    assert pending_document.processing_error is not None
    assert "S3 connection timeout" in pending_document.processing_error


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ingest_document_empty_chunks_still_ready(
    ingestion_session: AsyncSession,
    pending_document: Document,
    _patch_session_factory,
) -> None:
    """Document with no extractable chunks still transitions to 'ready'."""
    doc_id = str(pending_document.id)

    with (
        patch("src.services.ingestion.s3_client.download_file", return_value=Path("/tmp/fake.pdf")),
        patch("src.services.ingestion.s3_client.upload_bytes", return_value=None),
        patch("src.services.ingestion.docling_parser.parse", return_value=MockParseResult()),
        patch("src.services.ingestion.tasks._export_artifacts", return_value=(b"{}", b"# md")),
        patch("src.services.ingestion.chunker.chunk_document", return_value=[]),
    ):
        from src.services.ingestion.tasks import _run_pipeline

        await _run_pipeline(doc_id)

    await ingestion_session.refresh(pending_document)

    assert pending_document.status == "ready"
    assert pending_document.page_count == 3
