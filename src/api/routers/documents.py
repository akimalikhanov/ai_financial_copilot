"""Documents API."""

from __future__ import annotations

import logging
from typing import cast
from uuid import uuid4

from celery import Task
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from src.api.deps import CurrentUserDep
from src.db import DbSessionDep
from src.repository import DocumentRepository
from src.schemas.documents import (
    DocumentFilterOptionsResponse,
    DocumentListItem,
    ListDocumentsResponse,
    UploadDocumentResponse,
)
from src.services.ingestion.s3_client import upload_pdf
from src.services.ingestion.tasks import ingest_document

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
    storage_key = await upload_pdf(
        user_id=current_user.id,
        doc_id=doc_id,
        filename=file.filename or "document.pdf",
        fileobj=file.file,
        content_length=file_size,
    )

    metadata: dict = {}
    if company:
        metadata["company"] = company
    if year:
        metadata["year"] = year
    if doc_type:
        metadata["type"] = doc_type

    repo = DocumentRepository(session)
    doc = await repo.create(
        user_id=current_user.id,
        original_filename=file.filename or "document.pdf",
        storage_key=storage_key,
        content_type=ALLOWED_CONTENT_TYPE,
        file_size_bytes=file_size,
        metadata=metadata if metadata else None,
    )

    # Commit the new document row before enqueueing the task.
    # Otherwise the worker can run before the transaction commits and "update 0 rows",
    # leaving the document stuck in `pending`.
    await session.commit()

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
