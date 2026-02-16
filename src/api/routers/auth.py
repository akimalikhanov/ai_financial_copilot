"""Auth routes: register, login, refresh, logout."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from uuid import UUID

import bcrypt
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Request, Response, status

from src.api.deps import CurrentUserDep
from src.db.connection import DbSessionDep
from src.repository.session_repository import SessionRepository
from src.repository.user_repository import UserRepository
from src.schemas.auth import LoginRequest, RegisterRequest, TokenResponse, UserResponse
from src.services.auth.jwt_service import (
    create_access_token,
    create_refresh_token,
    decode_token,
    get_access_token_expire_seconds,
    get_refresh_cookie_max_age,
)

load_dotenv()

router = APIRouter(prefix="/v1/auth", tags=["auth"])

REFRESH_COOKIE = "refresh_token"
COOKIE_PATH = "/v1/auth"
COOKIE_SECURE = (os.getenv("COOKIE_SECURE") or "true").lower() == "true"


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def _verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode(), password_hash.encode())


def _set_refresh_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=REFRESH_COOKIE,
        value=token,
        max_age=get_refresh_cookie_max_age(),
        path=COOKIE_PATH,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="lax",
    )


def _clear_refresh_cookie(response: Response) -> None:
    """Clear refresh cookie; must match set_cookie params for browser to accept."""
    response.set_cookie(
        key=REFRESH_COOKIE,
        value="",
        max_age=0,
        path=COOKIE_PATH,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="lax",
    )


@router.post("/register", response_model=TokenResponse)
async def register(
    request: RegisterRequest,
    session: DbSessionDep,
    response: Response,
) -> TokenResponse:
    """Create user, return access_token in JSON, set refresh cookie."""
    user_repo = UserRepository(session)
    existing = await user_repo.get_by_email(request.email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )
    password_hash = _hash_password(request.password)
    user = await user_repo.create(
        email=request.email,
        password_hash=password_hash,
        display_name=request.display_name,
    )
    expires_at = datetime.now(UTC) + timedelta(seconds=get_refresh_cookie_max_age())
    session_repo = SessionRepository(session)
    sess = await session_repo.create(user.id, expires_at)
    access_token = create_access_token(user.id)
    refresh_token = create_refresh_token(user.id, sess.id)
    _set_refresh_cookie(response, refresh_token)
    return TokenResponse(
        access_token=access_token,
        expires_in=get_access_token_expire_seconds(),
    )


@router.post("/login", response_model=TokenResponse)
async def login(
    request: LoginRequest,
    session: DbSessionDep,
    response: Response,
) -> TokenResponse:
    """Verify credentials, return access_token in JSON, set refresh cookie."""
    user_repo = UserRepository(session)
    user = await user_repo.get_by_email(request.email)
    if (
        not user
        or not user.password_hash
        or not _verify_password(request.password, user.password_hash)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    expires_at = datetime.now(UTC) + timedelta(seconds=get_refresh_cookie_max_age())
    session_repo = SessionRepository(session)
    sess = await session_repo.create(user.id, expires_at)
    access_token = create_access_token(user.id)
    refresh_token = create_refresh_token(user.id, sess.id)
    _set_refresh_cookie(response, refresh_token)
    return TokenResponse(
        access_token=access_token,
        expires_in=get_access_token_expire_seconds(),
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh(
    http_request: Request,
    session: DbSessionDep,
    response: Response,
) -> TokenResponse:
    """Read refresh token from cookie; validate session; return new access_token."""
    token = http_request.cookies.get(REFRESH_COOKIE)
    if not token:
        _clear_refresh_cookie(response)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing refresh token"
        )
    payload = decode_token(token)
    if not payload or payload.get("type") != "refresh":
        _clear_refresh_cookie(response)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token"
        )
    user_id_str = payload.get("sub")
    jti = payload.get("jti")
    if not user_id_str or not jti:
        _clear_refresh_cookie(response)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token"
        )
    try:
        user_id = UUID(user_id_str)
        session_id = UUID(jti)
    except ValueError:
        _clear_refresh_cookie(response)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token"
        ) from None
    session_repo = SessionRepository(session)
    sess = await session_repo.get_valid_by_id(session_id)
    if not sess:
        _clear_refresh_cookie(response)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired or revoked"
        )
    access_token = create_access_token(user_id)
    return TokenResponse(
        access_token=access_token,
        expires_in=get_access_token_expire_seconds(),
    )


@router.get("/me", response_model=UserResponse)
async def me(current_user: CurrentUserDep) -> UserResponse:
    """Return current user (requires Bearer token)."""
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        display_name=current_user.display_name,
    )


@router.post("/logout")
async def logout(
    http_request: Request,
    session: DbSessionDep,
    response: Response,
) -> dict[str, str]:
    """Revoke session in DB and clear refresh cookie."""
    token = http_request.cookies.get(REFRESH_COOKIE)
    if token:
        payload = decode_token(token)
        if payload and payload.get("type") == "refresh" and payload.get("jti"):
            try:
                session_id = UUID(payload["jti"])
                session_repo = SessionRepository(session)
                await session_repo.revoke(session_id)
            except (ValueError, TypeError):
                pass
    _clear_refresh_cookie(response)
    return {"status": "ok"}
