from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status

from src.api.deps import CurrentUserDep
from src.db.connection import DbSessionDep
from src.models.user import User
from src.repository.conversation_repository import ConversationRepository
from src.repository.message_repository import MessageRepository
from src.schemas import chat as schemas

router = APIRouter(prefix="/v1/conversations", tags=["conversations"])


def _require_owner(conversation_user_id: UUID | None, current_user: User) -> None:
    if conversation_user_id is None or conversation_user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )


@router.post("", response_model=schemas.CreateConversationResponse)
async def create_conversation(
    request: schemas.CreateConversationRequest,
    session: DbSessionDep,
    current_user: CurrentUserDep,
) -> schemas.CreateConversationResponse:
    """Create a new conversation."""
    repo = ConversationRepository(session)
    conversation = await repo.create(
        user_id=current_user.id,
        title=request.title,
        settings=request.settings,
    )
    return schemas.CreateConversationResponse(conversation_id=conversation.id)


@router.patch("/{conversation_id}")
async def update_conversation(
    conversation_id: str,
    request: schemas.UpdateConversationRequest,
    session: DbSessionDep,
    current_user: CurrentUserDep,
) -> dict[str, str]:
    """Update a conversation."""
    try:
        conv_uuid = UUID(conversation_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid conversation_id format",
        ) from None

    repo = ConversationRepository(session)
    conversation = await repo.get_by_id(conv_uuid)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )
    _require_owner(conversation.user_id, current_user)
    await repo.update(conv_uuid, title=request.title)
    return {"status": "updated"}


@router.get("/{conversation_id}/messages")
async def get_messages(
    conversation_id: str,
    session: DbSessionDep,
    current_user: CurrentUserDep,
    limit: int = Query(default=50, ge=1, le=500),
    after_seq: int | None = Query(default=None, ge=0),
    before_seq: int | None = Query(default=None, ge=0),
) -> dict:
    """Get messages for a conversation with pagination."""
    try:
        conv_uuid = UUID(conversation_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid conversation_id format",
        ) from None

    conv_repo = ConversationRepository(session)
    conversation = await conv_repo.get_by_id(conv_uuid)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )
    _require_owner(conversation.user_id, current_user)

    # Get messages with pagination
    msg_repo = MessageRepository(session)
    if after_seq is not None:
        # Incremental fetch: get messages after seq
        messages = await msg_repo.get_after_seq(conv_uuid, after_seq, limit)
        has_more = len(messages) == limit
    elif before_seq is not None:
        # Paginated fetch: get recent messages before seq
        messages = await msg_repo.get_recent(conv_uuid, limit + 1, before_seq)
        has_more = len(messages) > limit
        if has_more:
            messages = messages[:-1]  # Remove extra message
    else:
        # Get recent messages (default: most recent first, then reverse)
        messages = await msg_repo.get_recent(conv_uuid, limit + 1)
        has_more = len(messages) > limit
        if has_more:
            messages = messages[:-1]  # Remove extra message

    # Convert to response format
    return {
        "messages": [
            {
                "id": str(msg.id),
                "role": msg.role.value,
                "content": msg.content,
                "seq": msg.seq,
                "created_at": msg.created_at.isoformat(),
                "metadata": msg.message_metadata,
            }
            for msg in messages
        ],
        "has_more": has_more,
    }
