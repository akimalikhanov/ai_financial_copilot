from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.message_feedback import FeedbackRating, MessageFeedback


class MessageFeedbackRepository:
    """Repository for message feedback (thumbs up/down) CRUD."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def upsert(
        self,
        message_id: UUID,
        user_id: UUID,
        rating: FeedbackRating,
        comment: str | None = None,
    ) -> MessageFeedback:
        """Insert or update feedback for (message, user) pair."""
        stmt = (
            insert(MessageFeedback)
            .values(
                message_id=message_id,
                user_id=user_id,
                rating=rating,
                comment=comment,
            )
            .on_conflict_do_update(
                index_elements=["message_id", "user_id"],
                set_={"rating": rating, "comment": comment},
            )
            .returning(MessageFeedback)
        )
        result = await self.session.execute(stmt)
        await self.session.flush()
        return result.scalar_one()

    async def get(self, message_id: UUID, user_id: UUID) -> MessageFeedback | None:
        result = await self.session.execute(
            select(MessageFeedback).where(
                MessageFeedback.message_id == message_id,
                MessageFeedback.user_id == user_id,
            )
        )
        return result.scalar_one_or_none()

    async def delete(self, message_id: UUID, user_id: UUID) -> bool:
        feedback = await self.get(message_id, user_id)
        if feedback is None:
            return False
        await self.session.delete(feedback)
        await self.session.flush()
        return True
