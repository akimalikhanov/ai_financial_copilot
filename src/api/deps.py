from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException, Request, status
from redis.asyncio import Redis

from src.db.connection import DbSessionDep
from src.models.user import User
from src.redis_client import check_chat_rate_limit
from src.repository.user_repository import UserRepository
from src.services.auth.jwt_service import decode_token
from src.services.llm_router import LLMRouter
from src.utils.config import get_rate_limit_retry_after_sec


def get_llm_router(request: Request) -> LLMRouter:
    """Retrieve LLMRouter from app state."""
    return request.app.state.llm_router


def get_redis(request: Request) -> Redis:
    """Retrieve Redis app client (rate limit, cache, SSE stream)."""
    return request.app.state.redis


async def get_current_user(request: Request, session: DbSessionDep) -> User:
    """Extract Bearer token, decode JWT, load user; raise 401 if invalid."""
    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header",
        )
    token = auth[7:].strip()
    payload = decode_token(token)
    if not payload or payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )
    sub = payload.get("sub")
    if not sub:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )
    try:
        user_id = UUID(sub)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        ) from None
    user_repo = UserRepository(session)
    user = await user_repo.get_by_id(user_id)
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    return user


async def chat_rate_limit(
    redis: Annotated[Redis, Depends(get_redis)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> None:
    """Rate limit chat enqueue per user. Raises 429 if over limit."""
    allowed, _ = await check_chat_rate_limit(redis, str(current_user.id))
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Retry later.",
            headers={"Retry-After": str(get_rate_limit_retry_after_sec())},
        )


# Type alias for dependency injection - use in route signatures
LLMRouterDep = Annotated[LLMRouter, Depends(get_llm_router)]
RedisDep = Annotated[Redis, Depends(get_redis)]
CurrentUserDep = Annotated[User, Depends(get_current_user)]
