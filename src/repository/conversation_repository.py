from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
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
        message_count: int,
    ) -> None:
        """Update conversation stats after message creation."""
        conversation = await self.get_by_id(conversation_id)
        if conversation:
            # Convert timezone-aware datetime to naive UTC for TIMESTAMP WITHOUT TIME ZONE
            now_utc = datetime.now(UTC)
            conversation.last_message_at = now_utc.replace(tzinfo=None)
            conversation.last_message_id = message_id
            conversation.message_count = message_count
            await self.session.flush()
