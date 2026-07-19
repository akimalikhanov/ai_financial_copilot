"""Tool schemas for the agent loop, generated from Pydantic arg models.

The Pydantic models are the single source of truth: their JSON schemas drive the
tool definitions handed to the LLM, and the same models parse the tool-call
arguments back. Schema and parser therefore cannot drift (P2-10, P0-3).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.schemas.agent_findings import AgentFindings, AnalyticalFindings
from src.utils.json_schema import make_strict


def tool_schema(name: str, description: str, args: type[BaseModel]) -> dict:
    """Build an OpenAI-compatible tool definition from a Pydantic arg model."""
    schema = args.model_json_schema()
    make_strict(schema)
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": schema},
    }


class SearchDocumentsArgs(BaseModel):
    entity: str = Field(description="The entity (company, fund, etc.) to search documents for.")
    query: str = Field(description="What to look for in that entity's documents.")


SEARCH_TOOL = tool_schema(
    "search_documents",
    "Search financial documents for a specific entity. Call once per entity.",
    SearchDocumentsArgs,
)

REPORT_FINDINGS_TOOL = tool_schema(
    "report_findings",
    "Call this once when you have finished searching. Report extracted values for all entities. This ends the search phase.",
    AgentFindings,
)

REPORT_ANALYTICAL_TOOL = tool_schema(
    "report_analytical_findings",
    "Call this once when you have a complete chain of observations for a causal or narrative question. This ends the search phase.",
    AnalyticalFindings,
)
