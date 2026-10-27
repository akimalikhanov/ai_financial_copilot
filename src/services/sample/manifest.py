"""Sample corpus manifest (SAMPLE_MANIFEST_PATH): the documents the demo user owns and the
questions offered in the UI."""

from __future__ import annotations

from functools import lru_cache
from uuid import UUID

import yaml
from pydantic import BaseModel, ConfigDict

from src.utils.config import get_sample_manifest_path


class SampleDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    file: str
    sha256: str
    title: str
    metadata: dict[str, str] = {}


class SampleQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    text: str


class SampleManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    documents: list[SampleDocument]
    questions: list[SampleQuestion]


@lru_cache(maxsize=1)
def load_manifest() -> SampleManifest:
    return SampleManifest.model_validate(yaml.safe_load(get_sample_manifest_path().read_text()))
