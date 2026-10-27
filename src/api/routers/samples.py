from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import CurrentUserDep, get_current_user
from src.db.connection import DbSessionDep
from src.repository import DocumentRepository
from src.repository.conversation_repository import ConversationRepository
from src.schemas.chat import CreateConversationResponse
from src.schemas.samples import SampleDocumentItem, SampleQuestionItem, SamplesResponse
from src.services.sample.corpus import CORPUS_KEY, SAMPLE_CORPUS
from src.services.sample.manifest import SampleManifest, load_manifest
from src.utils.config import get_demo_user_id

router = APIRouter(prefix="/v1/samples", tags=["samples"], dependencies=[Depends(get_current_user)])


async def _ready_manifest(session: AsyncSession) -> SampleManifest | None:
    """The manifest, when every document in it is ingested under the demo user."""
    demo_id = get_demo_user_id()
    if demo_id is None:
        return None
    manifest = load_manifest()
    docs = await DocumentRepository(session).get_by_ids([d.id for d in manifest.documents])
    ready = {d.id for d in docs if d.user_id == demo_id and d.status == "ready"}
    if not manifest.documents or any(d.id not in ready for d in manifest.documents):
        return None
    return manifest


@router.get("", response_model=SamplesResponse)
async def list_samples(session: DbSessionDep) -> SamplesResponse:
    manifest = await _ready_manifest(session)
    if manifest is None:
        return SamplesResponse()
    return SamplesResponse(
        documents=[SampleDocumentItem(id=d.id, title=d.title) for d in manifest.documents],
        questions=[SampleQuestionItem(id=q.id, text=q.text) for q in manifest.questions],
    )


@router.post("/conversations", response_model=CreateConversationResponse)
async def create_sample_conversation(
    session: DbSessionDep, current_user: CurrentUserDep
) -> CreateConversationResponse:
    """A conversation for the current user that searches the sample corpus."""
    manifest = await _ready_manifest(session)
    if manifest is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Sample documents are not available")
    conversation = await ConversationRepository(session).create(
        user_id=current_user.id,
        title=f"Sample: {manifest.documents[0].title}",
        metadata={CORPUS_KEY: SAMPLE_CORPUS},
    )
    return CreateConversationResponse(conversation_id=conversation.id)
