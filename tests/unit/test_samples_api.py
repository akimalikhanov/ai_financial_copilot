"""Samples API, sample-PDF access, and the committed manifest."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.routers import documents, samples
from src.models.user import User
from src.services.sample.manifest import load_manifest
from src.utils.config import get_sample_manifest_path

DEMO = uuid4()
USER = cast(User, SimpleNamespace(id=uuid4()))
SESSION = cast(AsyncSession, object())
DOC_ID = load_manifest().documents[0].id


@pytest.fixture(autouse=True)
def _demo_user(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEMO_USER_ID", str(DEMO))


def _docs(monkeypatch: pytest.MonkeyPatch, *docs: Any) -> None:
    class FakeRepo:
        def __init__(self, _session: Any) -> None:
            pass

        async def get_by_ids(self, _ids: list[UUID]) -> list[Any]:
            return list(docs)

        async def get_by_id(self, doc_id: UUID) -> Any:
            return next((d for d in docs if d.id == doc_id), None)

    monkeypatch.setattr(samples, "DocumentRepository", FakeRepo)
    monkeypatch.setattr(documents, "DocumentRepository", FakeRepo)


def _doc(user_id: UUID = DEMO, status: str = "ready", doc_id: UUID = DOC_ID) -> Any:
    return SimpleNamespace(id=doc_id, user_id=user_id, status=status, storage_key="k")


def test_manifest_matches_committed_pdf() -> None:
    for doc in load_manifest().documents:
        pdf = get_sample_manifest_path().parent / doc.file
        assert hashlib.sha256(pdf.read_bytes()).hexdigest() == doc.sha256


@pytest.mark.parametrize(
    "docs",
    [[], [_doc(status="processing")], [_doc(user_id=uuid4())]],
    ids=["missing", "not-ready", "wrong-owner"],
)
async def test_samples_empty_unless_ready(monkeypatch: pytest.MonkeyPatch, docs: list) -> None:
    _docs(monkeypatch, *docs)
    resp = await samples.list_samples(SESSION)
    assert resp.documents == [] and resp.questions == []
    with pytest.raises(HTTPException) as exc:
        await samples.create_sample_conversation(SESSION, USER)
    assert exc.value.status_code == 404


async def test_samples_disabled_without_demo_user(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEMO_USER_ID")
    _docs(monkeypatch, _doc())
    assert (await samples.list_samples(SESSION)).questions == []


async def test_samples_listed_when_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    _docs(monkeypatch, _doc())
    resp = await samples.list_samples(SESSION)
    assert [d.id for d in resp.documents] == [DOC_ID]
    assert len(resp.questions) == len(load_manifest().questions)


async def test_sample_conversation_is_flagged_server_side(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _docs(monkeypatch, _doc())
    created: dict[str, Any] = {}

    class FakeConvRepo:
        def __init__(self, _session: Any) -> None:
            pass

        async def create(self, **kwargs: Any) -> Any:
            created.update(kwargs)
            return SimpleNamespace(id=uuid4())

    monkeypatch.setattr(samples, "ConversationRepository", FakeConvRepo)
    await samples.create_sample_conversation(SESSION, USER)
    assert created["user_id"] == USER.id
    assert created["metadata"] == {"corpus": "demo"}


class _S3Reached(Exception):
    pass


@pytest.mark.parametrize(
    ("owner", "allowed"),
    [(USER.id, True), (DEMO, True), (uuid4(), False)],
    ids=["own", "demo", "other-user"],
)
async def test_pdf_access(monkeypatch: pytest.MonkeyPatch, owner: UUID, allowed: bool) -> None:
    _docs(monkeypatch, _doc(user_id=owner))

    def _session() -> Any:
        raise _S3Reached

    monkeypatch.setattr(documents.aioboto3, "Session", _session)
    expected: type[Exception] = _S3Reached if allowed else HTTPException
    with pytest.raises(expected):
        await documents.serve_pdf(DOC_ID, SESSION, USER)
