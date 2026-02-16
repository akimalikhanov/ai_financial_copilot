from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.conversation import Conversation


class ConversationRepository:
    """Repository for conversation CRUD operations."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        user_id: UUID | None = None,
        title: str | None = None,
        settings: dict | None = None,
        metadata: dict | None = None,
    ) -> Conversation:
        """Create a new conversation."""
        conversation = Conversation(
            user_id=user_id,
            title=title,
            settings=settings or {},
            conversation_metadata=metadata or {},
        )
        self.session.add(conversation)
        await self.session.flush()
        return conversation

    async def list_by_user(
        self, user_id: UUID, limit: int = 50, offset: int = 0
    ) -> tuple[list[Conversation], int]:
        """List non-deleted conversations for a user, ordered by activity."""
        count_result = await self.session.execute(
            select(func.count()).select_from(Conversation).where(
                Conversation.user_id == user_id,
                Conversation.deleted_at.is_(None),
            )
        )
        total = count_result.scalar() or 0

        result = await self.session.execute(
            select(Conversation)
            .where(
                Conversation.user_id == user_id,
                Conversation.deleted_at.is_(None),
            )
            .order_by(
                func.coalesce(
                    Conversation.last_message_at, Conversation.created_at
                ).desc()
            )
            .limit(limit)
            .offset(offset)
        )
        conversations = list(result.scalars().all())
        return conversations, total

    async def get_by_id(self, conversation_id: UUID) -> Conversation | None:
        """Get conversation by ID."""
        result = await self.session.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )
        return result.scalar_one_or_none()

    async def update(
        self,
        conversation_id: UUID,
        title: str | None = None,
    ) -> Conversation | None:
        """Update conversation title."""
        conversation = await self.get_by_id(conversation_id)
        if conversation:
            if title is not None:
                conversation.title = title
            await self.session.flush()
        return conversation

    async def update_on_message(
        self,
        conversation_id: UUID,
        message_id: UUID,
        new_seq: int,
    ) -> bool:
        """Update conversation stats after message creation with WHERE guard to prevent hot-spot updates."""
        # Convert timezone-aware datetime to naive UTC for TIMESTAMP WITHOUT TIME ZONE
        now_utc = datetime.now(UTC).replace(tzinfo=None)

        # Use WHERE guard to only update if new_seq > last_seq (prevents race conditions)
        result = await self.session.execute(
            update(Conversation)
            .where(
                Conversation.id == conversation_id,
                (Conversation.last_seq.is_(None)) | (Conversation.last_seq < new_seq),
            )
            .values(
                last_message_at=now_utc,
                last_message_id=message_id,
                last_seq=new_seq,
            )
        )
        await self.session.flush()
        return result.rowcount > 0
