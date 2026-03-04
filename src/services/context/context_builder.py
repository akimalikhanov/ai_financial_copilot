from __future__ import annotations

from uuid import UUID

from src.models.message import Message
from src.repository.message_repository import MessageRepository
from src.schemas import chat as schemas


def _db_message_to_chat_message(message: Message) -> schemas.ChatMessage:
    """Convert database Message model to ChatMessage schema."""
    from src.models.message import MessageRole

    role_map = {
        MessageRole.system: schemas.Role.system,
        MessageRole.user: schemas.Role.user,
        MessageRole.assistant: schemas.Role.assistant,
        MessageRole.tool: schemas.Role.tool,
    }
    return schemas.ChatMessage(
        role=role_map[message.role],
        content=message.content,
    )


async def build_context(
    message_repo: MessageRepository,
    conversation_id: UUID,
    before_seq: int | None = None,
    max_messages: int = 50,
) -> tuple[list[schemas.ChatMessage], int]:
    """
    Build context messages for LLM request with sliding window optimization.

    For long conversations, uses a sliding window approach to fetch only
    the most recent messages, improving performance and reducing token usage.

    Args:
        message_repo: Message repository instance
        conversation_id: Conversation ID
        before_seq: If provided, exclude messages with seq >= this (e.g. assistant placeholder)
        max_messages: Maximum number of messages to include (sliding window)

    Returns:
        Tuple of (messages, latest_seq). latest_seq is the last message's seq, or 0 if empty.
    """
    db_messages = await message_repo.get_recent(
        conversation_id, max_messages, before_seq=before_seq
    )
    messages = [_db_message_to_chat_message(msg) for msg in db_messages]
    latest_seq = db_messages[-1].seq if db_messages else 0
    return messages, latest_seq
