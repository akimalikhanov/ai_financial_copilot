"""Unit tests: User A cannot access User B's conversation."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.deps import get_current_user
from src.api.routers import conversations
from src.db.connection import get_db_session
from src.models.user import User
from src.models.conversation import Conversation


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(conversations.router)
    return app


async def _mock_get_db():
    yield MagicMock()


@pytest.fixture
def client() -> TestClient:
    c = TestClient(_app())
    c.app.dependency_overrides[get_db_session] = _mock_get_db
    yield c
    c.app.dependency_overrides.clear()


@pytest.fixture
def user_a() -> User:
    return User(
        id=uuid4(),
        email="a@test.com",
        display_name=None,
        auth_provider="local",
        auth_subject=None,
        password_hash=None,
        email_verified_at=None,
        is_active=True,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        last_seen_at=None,
        metadata_={},
    )


@pytest.fixture
def user_b_id():
    return uuid4()


def test_get_messages_returns_404_when_conversation_owned_by_other_user(
    client: TestClient,
    user_a: User,
    user_b_id,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """User A must not see messages of a conversation owned by User B."""
    conv_id = uuid4()
    conv = Conversation(
        id=conv_id,
        user_id=user_b_id,
        title="B's conversation",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        deleted_at=None,
        last_message_at=None,
        settings={},
        conversation_metadata={},
    )
    mock_conv_repo = MagicMock()
    mock_conv_repo.get_by_id = AsyncMock(return_value=conv)
    mock_msg_repo = MagicMock()

    def _fake_conv_repo(session: object) -> MagicMock:
        return mock_conv_repo

    def _fake_msg_repo(session: object) -> MagicMock:
        return mock_msg_repo

    monkeypatch.setattr("src.api.routers.conversations.ConversationRepository", _fake_conv_repo)
    monkeypatch.setattr("src.api.routers.conversations.MessageRepository", _fake_msg_repo)

    async def _override_current_user() -> User:
        return user_a

    client.app.dependency_overrides[get_current_user] = _override_current_user
    try:
        resp = client.get(f"/v1/conversations/{conv_id}/messages")
        assert resp.status_code == 404
        assert resp.json().get("detail") == "Conversation not found"
    finally:
        del client.app.dependency_overrides[get_current_user]
