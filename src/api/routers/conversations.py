from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException, status

from src.db.connection import DbSessionDep
from src.models.message import MessageRole
from src.repository.conversation_repository import ConversationRepository
from src.repository.message_repository import MessageRepository
from src.schemas import chat as schemas

router = APIRouter(prefix="/v1/conversations", tags=["conversations"])


@router.post("", response_model=schemas.CreateConversationResponse)
async def create_conversation(
    request: schemas.CreateConversationRequest,
    session: DbSessionDep,
) -> schemas.CreateConversationResponse:
    """Create a new conversation."""
    repo = ConversationRepository(session)
    conversation = await repo.create(
        user_id=request.user_id,
        title=request.title,
        settings=request.settings,
    )
    return schemas.CreateConversationResponse(conversation_id=conversation.id)


@router.patch("/{conversation_id}")
async def update_conversation(
    conversation_id: str,
    request: schemas.UpdateConversationRequest,
    session: DbSessionDep,
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
    conversation = await repo.update(conv_uuid, title=request.title)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    return {"status": "updated"}


@router.post("/{conversation_id}/messages", response_model=schemas.CreateMessageResponse)
async def create_message(
    conversation_id: str,
    request: schemas.CreateMessageBody,
    session: DbSessionDep,
) -> schemas.CreateMessageResponse:
    """Create a new message in a conversation."""
    try:
        conv_uuid = UUID(conversation_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid conversation_id format",
        ) from None

    # Verify conversation exists
    conv_repo = ConversationRepository(session)
    conversation = await conv_repo.get_by_id(conv_uuid)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    # Create message
    msg_repo = MessageRepository(session)
    # Convert schema Role to model MessageRole
    role_map = {
        schemas.Role.system: MessageRole.system,
        schemas.Role.user: MessageRole.user,
        schemas.Role.assistant: MessageRole.assistant,
        schemas.Role.tool: MessageRole.tool,
    }
    message_role = role_map.get(request.role, MessageRole.user)

    message = await msg_repo.create(
        conversation_id=conv_uuid,
        role=message_role,
        content=request.content,
        user_id=conversation.user_id,
        metadata=request.metadata,
    )

    # Update conversation stats
    await conv_repo.update_on_message(
        conversation_id=conv_uuid,
        message_id=message.id,
        message_count=message.seq,
    )

    return schemas.CreateMessageResponse(message_id=message.id, seq=message.seq)


@router.get("/{conversation_id}/messages")
async def get_messages(
    conversation_id: str,
    session: DbSessionDep,
) -> list[dict]:
    """Get all messages for a conversation."""
    try:
        conv_uuid = UUID(conversation_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid conversation_id format",
        ) from None

    # Verify conversation exists
    conv_repo = ConversationRepository(session)
    conversation = await conv_repo.get_by_id(conv_uuid)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    # Get messages
    msg_repo = MessageRepository(session)
    messages = await msg_repo.get_by_conversation_id(conv_uuid)

    # Convert to response format
    return [
        {
            "id": str(msg.id),
            "role": msg.role.value,
            "content": msg.content,
            "seq": msg.seq,
            "created_at": msg.created_at.isoformat(),
            "metadata": msg.message_metadata,
        }
        for msg in messages
    ]
