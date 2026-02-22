"""Schemas for documents API."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class UploadDocumentResponse(BaseModel):
    """Response from POST /v1/documents/upload."""

    model_config = ConfigDict(extra="forbid")

    id: UUID
    status: str
    original_filename: str
    storage_key: str
    created_at: datetime
    metadata: dict


class DocumentListItem(BaseModel):
    """Document returned from GET /v1/documents."""

    model_config = ConfigDict(extra="forbid")

    id: UUID
    status: str
    original_filename: str
    created_at: datetime
    extracted_title: str | None
    page_count: int | None
    metadata: dict


class ListDocumentsResponse(BaseModel):
    """Response from GET /v1/documents."""

    model_config = ConfigDict(extra="forbid")

    documents: list[DocumentListItem]
    total: int
