"""Whose documents a conversation searches: the requester's, or the demo user's for a
sample conversation. The only place that knows this rule."""

from __future__ import annotations

from uuid import UUID

from src.models.conversation import Conversation
from src.utils.config import get_demo_user_id

# conversations.metadata entry marking a sample conversation. Set server-side only.
CORPUS_KEY = "corpus"
SAMPLE_CORPUS = "demo"


def is_sample(conversation: Conversation) -> bool:
    meta = conversation.conversation_metadata
    return isinstance(meta, dict) and meta.get(CORPUS_KEY) == SAMPLE_CORPUS


def corpus_owner_id(conversation: Conversation, requester_id: UUID) -> UUID:
    """Raises if a sample conversation exists while the feature is disabled, rather than
    silently answering it from the requester's own documents."""
    if not is_sample(conversation):
        return requester_id
    demo_id = get_demo_user_id()
    if demo_id is None:
        raise RuntimeError("Sample conversation but DEMO_USER_ID is not set")
    return demo_id
