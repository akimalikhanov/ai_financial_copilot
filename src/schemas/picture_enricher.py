from __future__ import annotations

from pydantic import BaseModel


class PictureDescriptionItem(BaseModel):
    picture_id: int
    description: str


class PictureDescriptionResponse(BaseModel):
    descriptions: list[PictureDescriptionItem]
