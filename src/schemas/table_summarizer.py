from __future__ import annotations

from pydantic import BaseModel


class TableSummaryItem(BaseModel):
    table_id: int
    summary: str


class TableSummaryResponse(BaseModel):
    summaries: list[TableSummaryItem]
