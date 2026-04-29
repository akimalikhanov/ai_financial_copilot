from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException, status

from src.api.deps import CurrentUserDep
from src.db.connection import DbSessionDep
from src.models.message_feedback import FeedbackRating
from src.observability import langfuse as lf_client
from src.repository.conversation_repository import ConversationRepository
from src.repository.message_feedback_repository import MessageFeedbackRepository
from src.repository.message_repository import MessageRepository
from src.schemas.feedback import FeedbackRequest, FeedbackResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/messages", tags=["feedback"])


def _post_langfuse_score(trace_id: str | None, rating: FeedbackRating, comment: str | None) -> None:
    if not trace_id:
        logger.info("langfuse.score_skipped", extra={"reason": "no_trace_id"})
        return
    client = lf_client.get_client()
    if client is None:
        logger.warning("langfuse.score_skipped", extra={"reason": "client_not_initialized"})
        return
    try:
        client.create_score(
            name="user_feedback",
            value=1.0 if rating == FeedbackRating.up else 0.0,
            trace_id=trace_id,
            data_type="NUMERIC",
            comment=comment,
        )
        client.flush()
        logger.info("langfuse.score_posted", extra={"trace_id": trace_id, "rating": rating.value})
    except Exception as exc:  # noqa: BLE001
        logger.warning("langfuse.score_failed", extra={"trace_id": trace_id, "error": str(exc)})


@router.post("/{message_id}/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    message_id: str,
    request: FeedbackRequest,
    session: DbSessionDep,
    current_user: CurrentUserDep,
) -> FeedbackResponse:
    try:
        msg_uuid = UUID(message_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid message_id"
        ) from None

    msg_repo = MessageRepository(session)
    message = await msg_repo.get_by_id(msg_uuid)
    if message is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Message not found")
    # Assistant placeholder messages have NULL user_id; verify ownership via conversation.
    if message.user_id != current_user.id:
        conv_repo = ConversationRepository(session)
        conv = await conv_repo.get_by_id(message.conversation_id)
        if conv is None or conv.user_id != current_user.id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Message not found")

    feedback_repo = MessageFeedbackRepository(session)
    rating = FeedbackRating(request.rating)
    feedback = await feedback_repo.upsert(
        message_id=msg_uuid,
        user_id=current_user.id,
        rating=rating,
        comment=request.comment,
    )

    _post_langfuse_score(message.trace_id, rating, request.comment)

    return FeedbackResponse(
        message_id=str(feedback.message_id),
        rating=feedback.rating.value,
        comment=feedback.comment,
    )


@router.delete("/{message_id}/feedback")
async def delete_feedback(
    message_id: str,
    session: DbSessionDep,
    current_user: CurrentUserDep,
) -> dict[str, str]:
    try:
        msg_uuid = UUID(message_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid message_id"
        ) from None
    msg_repo = MessageRepository(session)
    message = await msg_repo.get_by_id(msg_uuid)
    if message is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Message not found")
    if message.user_id != current_user.id:
        conv_repo = ConversationRepository(session)
        conv = await conv_repo.get_by_id(message.conversation_id)
        if conv is None or conv.user_id != current_user.id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Message not found")

    feedback_repo = MessageFeedbackRepository(session)
    deleted = await feedback_repo.delete(msg_uuid, current_user.id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Feedback not found")
    return {"status": "deleted"}
