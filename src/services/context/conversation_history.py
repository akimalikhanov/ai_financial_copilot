from __future__ import annotations

import logging
from uuid import UUID

from redis.asyncio import Redis

from src.models.message import Message, MessageRole
from src.redis_client import (
    append_chat_tail,
    cas_populate_chat_tail,
    get_chat_tail,
    invalidate_chat_tail,
)
from src.repository.message_repository import MessageRepository
from src.schemas import chat as schemas
from src.utils.config import get_chat_tail_max_messages, get_chat_tail_ttl

logger = logging.getLogger(__name__)


def _db_message_to_chat_message(message: Message) -> schemas.ChatMessage:
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


class ConversationHistory:
    def __init__(
        self,
        redis: Redis,
        message_repo: MessageRepository,
        max_messages: int | None = None,
        ttl: int | None = None,
    ) -> None:
        self._redis = redis
        self._message_repo = message_repo
        self._max_messages = (
            max_messages if max_messages is not None else get_chat_tail_max_messages()
        )
        self._ttl = ttl if ttl is not None else get_chat_tail_ttl()

    async def load(
        self,
        conversation_id: UUID,
        before_seq: int | None,
        snapshot_seq: int,
    ) -> list[schemas.ChatMessage]:
        conv_id = str(conversation_id)
        cached = await get_chat_tail(self._redis, conv_id)
        if cached is not None:
            cached_msgs, cached_seq = cached
            if cached_seq >= snapshot_seq:
                try:
                    return [schemas.ChatMessage.model_validate(m) for m in cached_msgs]
                except Exception:
                    logger.warning(
                        "invalid_chat_tail_cache",
                        extra={"conversation_id": conv_id},
                    )

        messages, latest_seq = await self._fetch_from_db(conversation_id, before_seq)
        await cas_populate_chat_tail(
            self._redis,
            conv_id,
            [m.model_dump(mode="json") for m in messages],
            latest_seq,
        )
        return messages

    async def append_user(self, conversation_id: UUID, content: str, seq: int) -> None:
        try:
            await append_chat_tail(
                self._redis,
                str(conversation_id),
                schemas.ChatMessage(role=schemas.Role.user, content=content).model_dump(
                    mode="json"
                ),
                seq,
            )
        except Exception:
            logger.warning(
                "chat_tail_append_user_failed",
                extra={"conversation_id": str(conversation_id)},
            )

    async def append_assistant(self, conversation_id: UUID, content: str, seq: int) -> None:
        await append_chat_tail(
            self._redis,
            str(conversation_id),
            schemas.ChatMessage(
                role=schemas.Role.assistant,
                content=content,
            ).model_dump(mode="json"),
            seq,
        )

    async def invalidate(self, conversation_id: UUID) -> None:
        await invalidate_chat_tail(self._redis, str(conversation_id))

    async def _fetch_from_db(
        self,
        conversation_id: UUID,
        before_seq: int | None,
    ) -> tuple[list[schemas.ChatMessage], int]:
        db_messages = await self._message_repo.get_recent(
            conversation_id,
            self._max_messages,
            before_seq=before_seq,
        )
        messages = [_db_message_to_chat_message(msg) for msg in db_messages]
        latest_seq = db_messages[-1].seq if db_messages else 0
        return messages, latest_seq
