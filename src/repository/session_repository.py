from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.session import Session


class SessionRepository:
    """Repository for session CRUD and revoke (refresh token invalidation)."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, user_id: UUID, expires_at: datetime) -> Session:
        """Create a new session."""
        s = Session(user_id=user_id, expires_at=expires_at)
        self.session.add(s)
        await self.session.flush()
        return s

    async def get_valid_by_id(self, session_id: UUID) -> Session | None:
        """Return session if it exists, is not revoked, and not expired."""
        now = datetime.now(UTC)
        result = await self.session.execute(
            select(Session).where(
                Session.id == session_id,
                Session.revoked_at.is_(None),
                Session.expires_at > now,
            )
        )
        return result.scalar_one_or_none()

    async def revoke(self, session_id: UUID) -> None:
        """Set revoked_at to now()."""
        now = datetime.now(UTC)
        await self.session.execute(
            update(Session).where(Session.id == session_id).values(revoked_at=now)
        )
        await self.session.flush()
