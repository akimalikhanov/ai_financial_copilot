"""`corpus_owner_id`: sample conversations search the demo corpus, everything else the
requester's."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

import pytest

from src.models.conversation import Conversation
from src.services.sample.corpus import corpus_owner_id, is_sample

REQUESTER = uuid4()
DEMO = uuid4()


def _conv(metadata: Any) -> Conversation:
    return cast(Conversation, SimpleNamespace(conversation_metadata=metadata))


@pytest.fixture(autouse=True)
def _demo_user(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEMO_USER_ID", str(DEMO))


@pytest.mark.parametrize(
    "metadata",
    [{}, None, "demo", {"corpus": "other"}, {"corpus": None}, {"corpus": ["demo"]}],
    ids=["empty", "none", "not-a-dict", "other-corpus", "null-corpus", "list-corpus"],
)
def test_non_sample_uses_requester(metadata: Any) -> None:
    assert not is_sample(_conv(metadata))
    assert corpus_owner_id(_conv(metadata), REQUESTER) == REQUESTER


def test_sample_uses_demo_user() -> None:
    conv = _conv({"corpus": "demo", "other": 1})
    assert is_sample(conv)
    assert corpus_owner_id(conv, REQUESTER) == DEMO


def test_sample_without_demo_user_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEMO_USER_ID")
    with pytest.raises(RuntimeError):
        corpus_owner_id(_conv({"corpus": "demo"}), REQUESTER)
