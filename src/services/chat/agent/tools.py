"""Tool schemas for the agent loop.

Pydantic arg models are the single source of truth: their JSON schemas drive the tool
definitions handed to the LLM, and the same models parse the tool-call arguments back
— schema and parser cannot drift (P2-10, P0-3).

Post-D3 there is no registry: no tool is terminal and no tool has gates, so the only
thing a caller ever needs is the schema list.

Each path names its own pool (10b step 7). A shape-invariant pool under a shape-varying
prompt is what let an extraction run see `report_analytical_findings` — a tool `v3_agent`
never names, whose payload lands on the wrong ledger kind. `run_loop` assigns prompt and
pool on one line so neither can be set without the other.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.schemas.agent_findings import AgentFindings, Observation
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
    """Parses every `search_documents` call, on both paths.

    `sub_question` stays optional here because this one model must accept the extraction
    pool's two-field payload as well as the analytical pool's three-field one. The
    *schemas* differ (below); the parser is deliberately the looser of the two, so a call
    from either pool round-trips through it.
    """

    entity: str = Field(description="The entity (company, fund, etc.) to search documents for.")
    query: str = Field(description="What to look for in that entity's documents.")
    sub_question: str | None = Field(
        default=None,
        description=(
            "The question this search is trying to answer, in plain words. A new "
            "sub_question opens a new aspect; reuse the exact wording to re-search an "
            "aspect you already opened."
        ),
    )


class _ExtractionSearchArgs(BaseModel):
    """Schema-only: the two-field search the extraction path advertises.

    A separate model rather than a nullable field because `make_strict` forces every
    property into `required` — offering `sub_question` to a path with no decomposition
    would oblige the model to emit a null for a concept `v3_agent` never explains.
    """

    entity: str = Field(description="The entity (company, fund, etc.) to search documents for.")
    query: str = Field(description="What to look for in that entity's documents.")


SEARCH_TOOL = tool_schema(
    "search_documents",
    "Search financial documents for a specific entity. Call once per entity.",
    _ExtractionSearchArgs,
)

SEARCH_ANALYTICAL_TOOL = tool_schema(
    "search_documents",
    "Search financial documents for one aspect of the question. "
    "Each call targets ONE aspect, not one entity.",
    SearchDocumentsArgs,
)

REPORT_FINDINGS_TOOL = tool_schema(
    "report_findings",
    "Report extracted values for entities you have finished searching. "
    "You may call this more than once, and may search in the same turn — "
    "report each entity as soon as its evidence settles.",
    AgentFindings,
)


class _AnalyticalReportArgs(BaseModel):
    """Schema-only: `AnalyticalFindings` without `gaps`.

    Same reason as `_ExtractionSearchArgs` — `make_strict` forces every property into
    `required`, so advertising `gaps` obliges the model to author one on every call. A
    model-written gap is an unkeyed string: it closes no aspect, so the loop kept
    searching an aspect the model had already declared dead, and it could contradict a
    later grounded finding with no way to retract it. Negatives now go through
    `Observation.substantiated`, which closes its key like any other entry. The field
    stays on `AnalyticalFindings` — `projection()` still emits the loop's own keyed gaps.
    """

    question: str
    observations: tuple[Observation, ...]
    conclusion: str | None = None


REPORT_ANALYTICAL_TOOL = tool_schema(
    "report_analytical_findings",
    "Report observations for aspects whose evidence has settled. "
    "You may call this more than once, and may search in the same turn — "
    "report each aspect as soon as you can, rather than saving them all for the end.",
    _AnalyticalReportArgs,
)

ANALYTICAL_TOOLS = [SEARCH_ANALYTICAL_TOOL, REPORT_ANALYTICAL_TOOL]
EXTRACTION_TOOLS = [SEARCH_TOOL, REPORT_FINDINGS_TOOL]

# The loop partitions each turn on this set, so it must name every report tool across
# *both* pools — a report tool missing here would be routed to the search path.
REPORT_TOOL_NAMES = frozenset({"report_findings", "report_analytical_findings"})
