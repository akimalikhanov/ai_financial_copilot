"""Unit tests for auth middleware: valid token → user; invalid/missing → 401."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.deps import CurrentUserDep
from src.db.connection import get_db_session
from src.models.user import User


def _app() -> FastAPI:
    app = FastAPI()

    @app.get("/me")
    def _me(current_user: CurrentUserDep) -> dict:
        return {"user_id": str(current_user.id), "email": current_user.email}

    return app


async def _mock_get_db():
    yield MagicMock()


@pytest.fixture
def client() -> TestClient:
    c = TestClient(_app())
    c.app.dependency_overrides[get_db_session] = _mock_get_db
    yield c
    c.app.dependency_overrides.clear()


def test_missing_authorization_returns_401(client: TestClient) -> None:
    resp = client.get("/me")
    assert resp.status_code == 401
    assert "detail" in resp.json()


def test_invalid_authorization_prefix_returns_401(client: TestClient) -> None:
    resp = client.get("/me", headers={"Authorization": "Basic xyz"})
    assert resp.status_code == 401


def test_valid_token_returns_user(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id = UUID("a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11")
    user = User(
        id=user_id,
        email="u@example.com",
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
    mock_repo = MagicMock()
    mock_repo.get_by_id = AsyncMock(return_value=user)
    monkeypatch.setattr("src.api.deps.UserRepository", lambda _: mock_repo)
    monkeypatch.setattr(
        "src.api.deps.decode_token",
        lambda _: {"sub": str(user_id), "type": "access"},
    )
    resp = client.get("/me", headers={"Authorization": "Bearer any-token"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"] == str(user_id)
    assert data["email"] == "u@example.com"


def test_decode_token_returns_none_returns_401(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.api.deps.decode_token", lambda _: None)
    resp = client.get("/me", headers={"Authorization": "Bearer bad-token"})
    assert resp.status_code == 401
