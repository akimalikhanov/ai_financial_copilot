"""Documents API."""

from __future__ import annotations

from typing import cast
from uuid import uuid4

from celery import Task
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from src.api.deps import CurrentUserDep
from src.db import DbSessionDep
from src.repository import DocumentRepository
from src.schemas.documents import DocumentListItem, ListDocumentsResponse, UploadDocumentResponse
from src.services.ingestion.celery_app import ingest_document
from src.services.ingestion.s3_client import upload_pdf

router = APIRouter(prefix="/v1/documents", tags=["documents"])

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_CONTENT_TYPE = "application/pdf"


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

    body = await file.read()
    if len(body) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds 50MB limit",
        )

    doc_id = uuid4()
    storage_key = await upload_pdf(
        user_id=current_user.id,
        doc_id=doc_id,
        filename=file.filename or "document.pdf",
        body=body,
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
        file_size_bytes=len(body),
        metadata=metadata if metadata else None,
    )

    cast(Task, ingest_document).delay(str(doc.id))

    return UploadDocumentResponse(
        id=doc.id,
        status=doc.status,
        original_filename=doc.original_filename,
        storage_key=doc.storage_key,
        created_at=doc.created_at,
        metadata=doc.document_metadata,
    )
