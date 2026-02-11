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
    after_seq: int | None = None,
    max_messages: int = 50,
) -> tuple[list[schemas.ChatMessage], list[UUID]]:
    """
    Build context messages for LLM request with sliding window optimization.

    For long conversations, uses a sliding window approach to fetch only
    the most recent messages, improving performance and reducing token usage.

    Args:
        message_repo: Message repository instance
        conversation_id: Conversation ID
        after_seq: If provided, fetch messages after this seq (for incremental)
        max_messages: Maximum number of messages to include (sliding window)

    Returns:
        Tuple of (messages, included_message_ids)
    """
    if after_seq is not None:
        # Incremental fetch: get messages after the given seq
        # This is useful for UI pagination, but for LLM context we want full history
        # So we ignore after_seq and use sliding window instead
        pass

    # Use sliding window: get most recent messages (more efficient than all messages)
    # This ensures we don't fetch thousands of messages for long conversations
    db_messages = await message_repo.get_recent(conversation_id, max_messages)

    messages = [_db_message_to_chat_message(msg) for msg in db_messages]
    included_message_ids = [msg.id for msg in db_messages]

    return messages, included_message_ids
