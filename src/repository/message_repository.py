from __future__ import annotations

from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.message import Message, MessageRole, MessageStatus


class MessageRepository:
    """Repository for message CRUD operations."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        conversation_id: UUID,
        role: MessageRole,
        content: str,
        user_id: UUID | None = None,
        metadata: dict | None = None,
        client_msg_id: str | None = None,
        status: MessageStatus = MessageStatus.completed,
        request_id: UUID | None = None,
    ) -> Message:
        """Create a new message with auto-incremented seq."""
        # Get next seq for this conversation
        result = await self.session.execute(
            select(func.coalesce(func.max(Message.seq), 0)).where(
                Message.conversation_id == conversation_id
            )
        )
        next_seq = result.scalar_one() + 1

        message = Message(
            conversation_id=conversation_id,
            user_id=user_id,
            role=role,
            content=content,
            seq=next_seq,
            metadata=metadata or {},
            client_msg_id=client_msg_id,
            status=status,
            request_id=request_id,
        )
        self.session.add(message)
        await self.session.flush()
        return message

    async def get_by_conversation_id(
        self,
        conversation_id: UUID,
    ) -> list[Message]:
        """Get all messages for a conversation, ordered by seq."""
        result = await self.session.execute(
            select(Message).where(Message.conversation_id == conversation_id).order_by(Message.seq)
        )
        return list(result.scalars().all())

    async def update_on_final(
        self,
        message_id: UUID,
        content: str,
        request_id: UUID | None = None,
    ) -> Message | None:
        """Update message content and status on stream completion."""
        result = await self.session.execute(select(Message).where(Message.id == message_id))
        message = result.scalar_one_or_none()
        if not message:
            return None

        message.content = content
        message.status = MessageStatus.completed
        if request_id is not None:
            message.request_id = request_id

        await self.session.flush()
        return message
