"""Tool schemas for the agent loop.

Pydantic arg models are the single source of truth: their JSON schemas drive the tool
definitions handed to the LLM, and the same models parse the tool-call arguments back
— schema and parser cannot drift.

There is no registry: no tool is terminal and no tool has gates, so the only thing a
caller ever needs is the schema list.

Every shape offers `report_findings`; only the search schema differs, picked by
`ShapeConfig.tools`. The pool is also the dispatch rule: a call to a tool outside the
turn's pool gets "tool not available" and is never parsed.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.schemas.agent_findings import FindingsReport
from src.utils.json_schema import make_strict


def tool_schema(name: str, description: str, args: type[BaseModel]) -> dict:
    """Build an OpenAI-compatible tool definition from a Pydantic arg model."""
    schema = args.model_json_schema()
    make_strict(schema)
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": schema},
    }


_ENTITY_DESC = "The entity (company, fund, etc.) to search documents for."
_QUERY_DESC = (
    "Short phrase in the filing's own wording, for semantic search. Not a question. "
    "No company name."
)
_KEYWORDS_DESC = (
    "3-8 terms likely to appear verbatim in the filing, for keyword search: metric "
    "names, filing synonyms (revenue / net sales), years, currency codes. No company "
    "name, no intent words (change, highest, compare)."
)
_SUB_QUESTION_DESC = (
    "The question this search is trying to answer, in plain words. A new "
    "sub_question opens a new aspect; reuse the exact wording to re-search an "
    "aspect you already opened."
)


class SearchDocumentsArgs(BaseModel):
    """Parses every `search_documents` call, on both paths.

    `keywords` and `sub_question` stay optional here so one parser accepts either pool's
    payload. The advertised *schemas* (below) are stricter; a call missing `keywords`
    still parses, and the loop searches BM25 with `query` instead.
    """

    entity: str = Field(description=_ENTITY_DESC)
    query: str = Field(description=_QUERY_DESC)
    keywords: str | None = Field(default=None, description=_KEYWORDS_DESC)
    sub_question: str | None = Field(default=None, description=_SUB_QUESTION_DESC)


class _ExtractionSearchArgs(BaseModel):
    """Schema-only: the search the extraction path advertises.

    Separate schema models rather than nullable fields on the parser because
    `make_strict` forces every property into `required` — a nullable `keywords` would let
    the model emit null, and `sub_question` would oblige the extraction model to emit a
    null for a concept `v4_agent` never explains.
    """

    entity: str = Field(description=_ENTITY_DESC)
    query: str = Field(description=_QUERY_DESC)
    keywords: str = Field(description=_KEYWORDS_DESC)


class _AnalyticalSearchArgs(_ExtractionSearchArgs):
    """Schema-only: the extraction search plus the aspect-minting `sub_question`."""

    sub_question: str | None = Field(default=None, description=_SUB_QUESTION_DESC)


SEARCH_TOOL = tool_schema(
    "search_documents",
    "Search financial documents for a specific entity. Call once per entity.",
    _ExtractionSearchArgs,
)

SEARCH_ANALYTICAL_TOOL = tool_schema(
    "search_documents",
    "Search financial documents for one aspect of the question. "
    "Each call targets ONE aspect, not one entity.",
    _AnalyticalSearchArgs,
)

REPORT_TOOL_NAME = "report_findings"

REPORT_FINDINGS_TOOL = tool_schema(
    REPORT_TOOL_NAME,
    "Report findings for keys whose evidence has settled. "
    "You may call this more than once, and may search in the same turn — "
    "report each key as soon as its evidence settles.",
    FindingsReport,
)
