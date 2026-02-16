from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.user import User


class UserRepository:
    """Repository for user CRUD (login lookup, JWT subject resolution, registration)."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_email(self, email: str) -> User | None:
        """Get user by email (for login lookup)."""
        result = await self.session.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()

    async def get_by_id(self, id: UUID) -> User | None:
        """Get user by id (for JWT subject resolution)."""
        result = await self.session.execute(select(User).where(User.id == id))
        return result.scalar_one_or_none()

    async def create(
        self,
        email: str,
        password_hash: str,
        display_name: str | None = None,
        auth_provider: str = "local",
    ) -> User:
        """Create a new user (for registration)."""
        user = User(
            email=email,
            password_hash=password_hash,
            display_name=display_name,
            auth_provider=auth_provider,
        )
        self.session.add(user)
        await self.session.flush()
        return user
