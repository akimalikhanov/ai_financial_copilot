from __future__ import annotations

from fastapi import APIRouter

from src.api.routers.chat import router as chat_router
from src.api.routers.models import router as models_router


def get_routers() -> list[APIRouter]:
    return [chat_router, models_router]
