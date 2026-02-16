"""Auth request/response schemas."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict


class RegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: str
    password: str
    display_name: str | None = None


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: str
    password: str


class TokenResponse(BaseModel):
    """Access token in JSON; refresh token is set via httpOnly cookie."""

    model_config = ConfigDict(extra="forbid")

    access_token: str
    token_type: str = "Bearer"
    expires_in: int  # seconds until expiration


class UserResponse(BaseModel):
    """User info for /me endpoint."""

    model_config = ConfigDict(extra="forbid")

    id: UUID
    email: str
    display_name: str | None = None
