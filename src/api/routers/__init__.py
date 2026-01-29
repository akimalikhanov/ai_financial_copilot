from __future__ import annotations

from fastapi import APIRouter

from src.api.routers.chat import router as chat_router


def get_routers() -> list[APIRouter]:
    return [chat_router]
