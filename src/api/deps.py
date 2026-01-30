from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from src.services.llm_router import LLMRouter


def get_llm_router(request: Request) -> LLMRouter:
    """Retrieve LLMRouter from app state."""
    return request.app.state.llm_router


# Type alias for dependency injection - use in route signatures
LLMRouterDep = Annotated[LLMRouter, Depends(get_llm_router)]
