from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class FeedbackRequest(BaseModel):
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=2000)


class FeedbackResponse(BaseModel):
    message_id: str
    rating: Literal["up", "down"]
    comment: str | None = None
