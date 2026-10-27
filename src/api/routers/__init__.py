from __future__ import annotations

from fastapi import APIRouter

from src.api.routers.auth import router as auth_router
from src.api.routers.chat import router as chat_router
from src.api.routers.conversations import router as conversations_router
from src.api.routers.documents import router as documents_router
from src.api.routers.feedback import router as feedback_router
from src.api.routers.models import router as models_router
from src.api.routers.samples import router as samples_router


def get_routers() -> list[APIRouter]:
    return [
        auth_router,
        chat_router,
        conversations_router,
        documents_router,
        feedback_router,
        models_router,
        samples_router,
    ]
