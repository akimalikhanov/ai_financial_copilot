from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PromptTemplate(BaseModel):
    """Schema for a prompt template loaded from YAML."""

    model_config = ConfigDict(extra="forbid")

    version: str = Field(..., description="Template version (e.g., 'v1', 'v2')")
    name: str = Field(..., description="Template name (e.g., 'system', 'persona', 'guidelines')")
    description: str | None = Field(None, description="Human-readable description of the template")
    template: str = Field(..., description="Jinja2 template string")
    variables: list[str] = Field(
        default_factory=list,
        description="List of required variable names for rendering",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (author, created_at, etc.)",
    )
