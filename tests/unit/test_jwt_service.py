"""Unit tests for JWT token creation, decoding, and expiry."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import jwt
import pytest

from src.services.auth import jwt_service


@pytest.fixture(autouse=True)
def _jwt_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-key-min-32-chars-long")
    monkeypatch.setenv("JWT_ALGORITHM", "HS256")


def test_create_access_token_contains_sub_and_type() -> None:
    user_id = uuid4()
    token = jwt_service.create_access_token(user_id)
    payload = jwt_service.decode_token(token)
    assert payload is not None
    assert payload["sub"] == str(user_id)
    assert payload["type"] == "access"
    assert "exp" in payload


def test_create_refresh_token_contains_jti_and_type() -> None:
    user_id = uuid4()
    session_id = uuid4()
    token = jwt_service.create_refresh_token(user_id, session_id)
    payload = jwt_service.decode_token(token)
    assert payload is not None
    assert payload["sub"] == str(user_id)
    assert payload["jti"] == str(session_id)
    assert payload["type"] == "refresh"
    assert "exp" in payload


def test_decode_token_invalid_returns_none() -> None:
    assert jwt_service.decode_token("invalid") is None
    assert jwt_service.decode_token("") is None


def test_decode_token_wrong_secret_returns_none() -> None:
    user_id = uuid4()
    token = jwt.encode(
        {"sub": str(user_id), "type": "access", "exp": datetime.now(timezone.utc) + timedelta(minutes=15)},
        "wrong-secret",
        algorithm="HS256",
    )
    assert jwt_service.decode_token(token) is None


def test_decode_token_expired_returns_none() -> None:
    user_id = uuid4()
    secret = "test-secret-key-min-32-chars-long"
    token = jwt.encode(
        {"sub": str(user_id), "type": "access", "exp": datetime.now(timezone.utc) - timedelta(minutes=1)},
        secret,
        algorithm="HS256",
    )
    assert jwt_service.decode_token(token) is None
