"""JWT token creation and validation for auth."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from uuid import UUID

from dotenv import load_dotenv
import jwt

load_dotenv()


def _get_secret() -> str:
    secret = os.getenv("JWT_SECRET_KEY") or os.getenv("JWT_SECRET")
    if not secret:
        raise RuntimeError("JWT_SECRET_KEY or JWT_SECRET is required")
    return secret


def _get_algorithm() -> str:
    return os.getenv("JWT_ALGORITHM")


def _parse_expire(value: str, default_minutes: int) -> int:
    """Parse expire string (e.g. '15m', '7d') into minutes."""
    if not value:
        return default_minutes
    value = value.strip().lower()
    if value.endswith("m"):
        return int(value[:-1] or default_minutes)
    if value.endswith("d"):
        return int(value[:-1] or 0) * 24 * 60
    return int(value) if value.isdigit() else default_minutes


def _access_expire_minutes() -> int:
    return _parse_expire(
        os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES") or os.getenv("JWT_EXPIRATION_TIME", ""),
        15,
    )


def _refresh_expire_minutes() -> int:
    raw = os.getenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS") or os.getenv("JWT_REFRESH_TIME", "")
    if raw and raw.isdigit():
        return int(raw) * 24 * 60
    return _parse_expire(raw, 7 * 24 * 60)


def create_access_token(user_id: UUID) -> str:
    """Create a short-lived access token."""
    exp = datetime.now(timezone.utc) + timedelta(minutes=_access_expire_minutes())
    payload = {"sub": str(user_id), "exp": exp, "type": "access"}
    return jwt.encode(payload, _get_secret(), algorithm=_get_algorithm())


def create_refresh_token(user_id: UUID, session_id: UUID) -> str:
    """Create a refresh token bound to a session (jti = session_id for revocation)."""
    exp = datetime.now(timezone.utc) + timedelta(minutes=_refresh_expire_minutes())
    payload = {"sub": str(user_id), "jti": str(session_id), "exp": exp, "type": "refresh"}
    return jwt.encode(payload, _get_secret(), algorithm=_get_algorithm())


def decode_token(token: str) -> dict | None:
    """Decode and validate a JWT; return payload or None if invalid."""
    try:
        return jwt.decode(
            token,
            _get_secret(),
            algorithms=[_get_algorithm()],
            options={"verify_exp": True},
        )
    except jwt.PyJWTError:
        return None
