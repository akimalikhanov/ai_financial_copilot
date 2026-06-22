"""Documents API."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncGenerator
from typing import cast
from uuid import UUID, uuid4

import aioboto3
from celery import Task
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy import text

from src.api.deps import CurrentUserDep, RedisDep
from src.api.exceptions import _sse_event
from src.db import DbSessionDep
from src.redis_client import ingestion_stream_key
from src.repository import DocumentRepository
from src.schemas.documents import (
    DocumentFilterOptionsResponse,
    DocumentListItem,
    ListDocumentsResponse,
    UploadDocumentResponse,
)
from src.services.ingestion import opensearch_ingest, qdrant_ingest
from src.services.ingestion.s3_client import build_raw_storage_key, upload_pdf
from src.services.ingestion.tasks import ingest_document
from src.utils.config import (
    get_s3_access_key,
    get_s3_chunks_bucket,
    get_s3_docling_bucket,
    get_s3_endpoint_url,
    get_s3_raw_bucket,
    get_s3_rendered_bucket,
    get_s3_secret_key,
)

router = APIRouter(prefix="/v1/documents", tags=["documents"])

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_CONTENT_TYPE = "application/pdf"

logger = logging.getLogger(__name__)


@router.get("", response_model=ListDocumentsResponse)
async def list_documents(
    session: DbSessionDep,
    current_user: CurrentUserDep,
) -> ListDocumentsResponse:
    repo = DocumentRepository(session)
    docs = await repo.list_by_user(current_user.id)
    items = [
        DocumentListItem(
            id=d.id,
            status=d.status,
            original_filename=d.original_filename,
            created_at=d.created_at,
            extracted_title=d.extracted_title,
            page_count=d.page_count,
            metadata=d.document_metadata,
        )
        for d in docs
    ]
    return ListDocumentsResponse(documents=items, total=len(items))


@router.get("/filter-options", response_model=DocumentFilterOptionsResponse)
async def get_filter_options(
    session: DbSessionDep,
    current_user: CurrentUserDep,
) -> DocumentFilterOptionsResponse:
    repo = DocumentRepository(session)
    options = await repo.get_filter_options(current_user.id)
    return DocumentFilterOptionsResponse(**options)


@router.post("/upload", response_model=UploadDocumentResponse)
async def upload_document(
    session: DbSessionDep,
    current_user: CurrentUserDep,
    file: UploadFile = File(...),
    company: str | None = Form(None),
    year: str | None = Form(None),
    doc_type: str | None = Form(None, alias="type"),
) -> UploadDocumentResponse:
    """
    Upload a PDF. Stores in S3, creates document record, enqueues ingestion job.
    """
    if file.content_type != ALLOWED_CONTENT_TYPE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are allowed",
        )

    try:
        pos = file.file.tell()
        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(pos)
    except Exception:
        file_size = None

    if file_size is not None and file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds {MAX_FILE_SIZE // (1024 * 1024)}MB limit",
        )

    doc_id = uuid4()
    filename = file.filename or "document.pdf"
    storage_key = build_raw_storage_key(current_user.id, doc_id, filename)

    metadata: dict = {}
    if company:
        metadata["company"] = company
    if year:
        metadata["year"] = year
    if doc_type:
        metadata["type"] = doc_type

    # Create the DB row before touching S3: the storage key is deterministic,
    # so if the upload fails we can just delete this row instead of leaving
    # an orphaned PDF in S3 with no DB record pointing to it.
    repo = DocumentRepository(session)
    doc = await repo.create(
        id=doc_id,
        user_id=current_user.id,
        original_filename=filename,
        storage_key=storage_key,
        content_type=ALLOWED_CONTENT_TYPE,
        file_size_bytes=file_size,
        metadata=metadata if metadata else None,
    )
    await session.commit()

    try:
        await upload_pdf(
            user_id=current_user.id,
            doc_id=doc_id,
            filename=filename,
            fileobj=file.file,
            content_length=file_size,
        )
    except Exception:
        await session.execute(text("DELETE FROM documents WHERE id = :id"), {"id": doc_id})
        await session.commit()
        logger.exception("document.upload_failed", extra={"document_id": str(doc_id)})
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to store document",
        ) from None

    cast(Task, ingest_document).delay(str(doc.id))
    logger.info("document.ingestion_enqueued", extra={"document_id": str(doc.id)})

    return UploadDocumentResponse(
        id=doc.id,
        status=doc.status,
        original_filename=doc.original_filename,
        storage_key=doc.storage_key,
        created_at=doc.created_at,
        metadata=doc.document_metadata,
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: UUID,
    session: DbSessionDep,
    current_user: CurrentUserDep,
) -> None:
    repo = DocumentRepository(session)
    doc = await repo.get_by_id(document_id)
    if doc is None or doc.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    qdrant_collection = os.getenv("QDRANT_COLLECTION", "documents")
    opensearch_index = os.getenv("OPENSEARCH_INDEX", "chunks")

    try:
        qdrant_ingest.delete_by_document(qdrant_collection, document_id)
    except Exception:
        logger.warning("delete_document.qdrant_failed", extra={"document_id": str(document_id)})

    try:
        opensearch_ingest.delete_by_document(opensearch_index, document_id)
    except Exception:
        logger.warning("delete_document.opensearch_failed", extra={"document_id": str(document_id)})

    s3_keys = {
        get_s3_raw_bucket(): [doc.storage_key],
        get_s3_docling_bucket(): [f"processed/{doc.user_id}/{document_id}/docling.json"],
        get_s3_rendered_bucket(): [f"processed/{doc.user_id}/{document_id}/document.md"],
        get_s3_chunks_bucket(): [f"processed/{doc.user_id}/{document_id}/chunks.jsonl"],
    }
    s3_session = aioboto3.Session()
    async with s3_session.client(  # type: ignore[attr-defined]
        "s3",
        endpoint_url=get_s3_endpoint_url(),
        region_name="garage",
        aws_access_key_id=get_s3_access_key(),
        aws_secret_access_key=get_s3_secret_key(),
    ) as s3:
        for bucket, keys in s3_keys.items():
            for key in keys:
                try:
                    await s3.delete_object(Bucket=bucket, Key=key)
                except Exception:
                    logger.warning(
                        "delete_document.s3_failed",
                        extra={"bucket": bucket, "key": key},
                    )

    await session.execute(text("DELETE FROM chunks WHERE document_id = :id"), {"id": document_id})
    await session.execute(text("DELETE FROM documents WHERE id = :id"), {"id": document_id})
    await session.commit()
    logger.info("document.deleted", extra={"document_id": str(document_id)})


@router.get("/{document_id}/stream")
async def ingestion_stream(
    document_id: UUID,
    request: Request,  # noqa: ARG001
    session: DbSessionDep,
    redis: RedisDep,
    current_user: CurrentUserDep,
) -> StreamingResponse:
    """SSE stream for ingestion progress of a document."""
    repo = DocumentRepository(session)
    doc = await repo.get_by_id(document_id)
    if doc is None or doc.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    # Already terminal — return a synthetic done/error immediately.
    if doc.status == "ready":

        async def _immediate_done() -> AsyncGenerator[str, None]:
            yield _sse_event("done", {})

        return StreamingResponse(
            _immediate_done(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    if doc.status == "failed":

        async def _immediate_error() -> AsyncGenerator[str, None]:
            yield _sse_event("error", {"message": doc.processing_error or "Ingestion failed"})

        return StreamingResponse(
            _immediate_error(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    stream_key = ingestion_stream_key(str(document_id))
    last_id = "0-0"

    async def event_stream() -> AsyncGenerator[str, None]:
        nonlocal last_id
        empty_polls = 0
        yield ": ok\n\n"
        try:
            while True:
                result = await redis.xread({stream_key: last_id}, block=15000, count=20)
                if not result:
                    empty_polls += 1
                    if empty_polls >= 4:
                        # Worker may have died — check DB status
                        fresh = await repo.get_by_id(document_id)
                        if fresh and fresh.status == "ready":
                            yield _sse_event("done", {})
                            return
                        if fresh and fresh.status == "failed":
                            yield _sse_event(
                                "error", {"message": fresh.processing_error or "Ingestion failed"}
                            )
                            return
                    yield ": keepalive\n\n"
                    continue

                empty_polls = 0
                for _, events in result:
                    for eid, raw_data in events:
                        last_id = eid
                        payload_str = (
                            raw_data.get("payload") if isinstance(raw_data, dict) else None
                        )
                        if not payload_str:
                            continue
                        try:
                            data = json.loads(payload_str)
                        except json.JSONDecodeError:
                            continue
                        event_type = data.get("type", "stage")
                        sse_data = {k: v for k, v in data.items() if k != "type"}
                        yield _sse_event(event_type, sse_data)
                        if event_type in ("done", "error"):
                            return
        except asyncio.CancelledError:
            raise
        except Exception:
            yield _sse_event("error", {"message": "Stream read failed"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/{document_id}/pdf")
async def serve_pdf(
    document_id: UUID,
    session: DbSessionDep,
    current_user: CurrentUserDep,
) -> StreamingResponse:
    repo = DocumentRepository(session)
    doc = await repo.get_by_id(document_id)
    if doc is None or doc.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    s3_session = aioboto3.Session()
    client = await s3_session.client(
        "s3",
        endpoint_url=get_s3_endpoint_url(),
        region_name="garage",
        aws_access_key_id=get_s3_access_key(),
        aws_secret_access_key=get_s3_secret_key(),
    ).__aenter__()

    try:
        resp = await client.get_object(Bucket=get_s3_raw_bucket(), Key=doc.storage_key)
    except Exception as err:
        await client.__aexit__(None, None, None)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="File not found in storage"
        ) from err

    content_length: int | None = resp.get("ContentLength")
    body = resp["Body"]

    async def _stream():
        try:
            async for chunk in body.iter_chunks(1024 * 256):
                yield chunk
        finally:
            await client.__aexit__(None, None, None)

    headers = {
        "Content-Disposition": f'inline; filename="{doc.original_filename}"',
        **({"Content-Length": str(content_length)} if content_length else {}),
    }
    return StreamingResponse(_stream(), media_type="application/pdf", headers=headers)
