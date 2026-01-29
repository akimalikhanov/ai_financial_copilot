from __future__ import annotations

from fastapi import Request

from src.services.llm_router import LLMRouter


def get_llm_router(request: Request) -> LLMRouter:
    return request.app.state.llm_router
