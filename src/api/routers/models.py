from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from src.api.deps import LLMRouterDep

router = APIRouter(prefix="/v1/models", tags=["models"])


class ModelInfo(BaseModel):
    id: str
    name: str


class ModelsResponse(BaseModel):
    models: list[ModelInfo]


@router.get("", response_model=ModelsResponse)
async def list_models(llm_router: LLMRouterDep) -> ModelsResponse:
    """Return list of available models with id and display name."""
    config = llm_router._config
    models_list: list[ModelInfo] = []

    for m in config.get("models") or []:
        model_id = m.get("id")
        label = m.get("label") or model_id
        if model_id:
            models_list.append(ModelInfo(id=model_id, name=label))

    return ModelsResponse(models=models_list)
