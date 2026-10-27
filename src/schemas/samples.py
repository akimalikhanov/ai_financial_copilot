"""Schemas for the samples API."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict


class SampleDocumentItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    title: str


class SampleQuestionItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    text: str


class SamplesResponse(BaseModel):
    """GET /v1/samples. Empty when the sample corpus isn't ready."""

    model_config = ConfigDict(extra="forbid")

    documents: list[SampleDocumentItem] = []
    questions: list[SampleQuestionItem] = []
