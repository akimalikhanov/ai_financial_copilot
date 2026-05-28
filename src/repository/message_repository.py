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
            message_metadata=metadata or {},
            client_msg_id=client_msg_id,
            status=status,
            request_id=request_id,
        )
        self.session.add(message)
        await self.session.flush()
        return message

    async def get_by_id(self, message_id: UUID) -> Message | None:
        """Get a single message by ID."""
        result = await self.session.execute(select(Message).where(Message.id == message_id))
        return result.scalar_one_or_none()

    async def get_by_client_msg_id(
        self, conversation_id: UUID, client_msg_id: str
    ) -> Message | None:
        """Get message by client_msg_id for idempotency check."""
        result = await self.session.execute(
            select(Message).where(
                Message.conversation_id == conversation_id,
                Message.client_msg_id == client_msg_id,
            )
        )
        return result.scalar_one_or_none()

    async def get_after_seq(
        self, conversation_id: UUID, after_seq: int, limit: int
    ) -> list[Message]:
        """Get messages after a given sequence number (incremental fetch)."""
        result = await self.session.execute(
            select(Message)
            .where(
                Message.conversation_id == conversation_id,
                Message.seq > after_seq,
            )
            .order_by(Message.seq)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_recent(
        self, conversation_id: UUID, limit: int, before_seq: int | None = None
    ) -> list[Message]:
        """Get recent messages (paginated, ordered by seq descending)."""
        query = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.seq.desc())
            .limit(limit)
        )
        if before_seq is not None:
            query = query.where(Message.seq < before_seq)

        result = await self.session.execute(query)
        messages = list(result.scalars().all())
        # Return in ascending order (oldest first)
        return list(reversed(messages))

    async def update_on_final(
        self,
        message_id: UUID,
        content: str,
        request_id: UUID | None = None,
        raw_content: str | None = None,
        metadata_updates: dict | None = None,
        trace: dict | None = None,
        trace_id: str | None = None,
        agent_findings: dict | None = None,
    ) -> Message | None:
        """Update message content and status on stream completion."""
        result = await self.session.execute(select(Message).where(Message.id == message_id))
        message = result.scalar_one_or_none()
        if not message:
            return None

        message.content = content
        message.status = MessageStatus.completed
        if raw_content is not None:
            message.raw_content = raw_content
        if request_id is not None:
            message.request_id = request_id
        if metadata_updates:
            message.message_metadata = {**message.message_metadata, **metadata_updates}
        if trace is not None:
            message.trace = trace
        if trace_id is not None:
            message.trace_id = trace_id
        if agent_findings is not None:
            message.agent_findings = agent_findings

        await self.session.flush()
        return message
