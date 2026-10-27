"""`plan_seed`: what the seed does with the demo user's documents."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID, uuid4

import pytest

from src.models.document import Document
from src.services.sample.manifest import SampleDocument, SampleManifest
from src.services.sample.seed import plan_seed

WANTED = uuid4()
MANIFEST = SampleManifest(
    documents=[SampleDocument(id=WANTED, file="a.pdf", sha256="x", title="A")], questions=[]
)


def _doc(status: str, doc_id: UUID = WANTED) -> Document:
    return cast(Document, SimpleNamespace(id=doc_id, status=status))


def _ids(items: list[Any]) -> list[UUID]:
    return [i.id for i in items]


def test_missing_is_created() -> None:
    plan = plan_seed([], MANIFEST)
    assert _ids(plan.create) == [WANTED]
    assert plan.purge == [] and plan.enqueue == [] and plan.processing == []


def test_ready_is_a_noop() -> None:
    plan = plan_seed([_doc("ready")], MANIFEST)
    assert plan.purge == [] and plan.create == [] and plan.enqueue == []


def test_failed_is_purged_and_recreated() -> None:
    plan = plan_seed([_doc("failed")], MANIFEST)
    assert _ids(plan.purge) == [WANTED]
    assert _ids(plan.create) == [WANTED]


def test_unlisted_is_purged() -> None:
    stale = uuid4()
    plan = plan_seed([_doc("ready", stale), _doc("ready")], MANIFEST)
    assert _ids(plan.purge) == [stale]
    assert plan.create == []


@pytest.mark.parametrize(
    ("status", "enqueued", "processing"),
    [("pending", True, False), ("processing", False, True)],
)
def test_in_flight(status: str, enqueued: bool, processing: bool) -> None:
    plan = plan_seed([_doc(status)], MANIFEST)
    assert plan.create == [] and plan.purge == []
    assert (plan.enqueue == [WANTED]) is enqueued
    assert (_ids(plan.processing) == [WANTED]) is processing
