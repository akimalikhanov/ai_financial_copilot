"""Tool schemas + registry for the agent loop.

Pydantic arg models are the single source of truth: their JSON schemas drive the tool
definitions handed to the LLM, and the same models parse the tool-call arguments back
— schema and parser cannot drift (P2-10, P0-3).

`TOOL_REGISTRY` replaces the old `_FINALIZER_NAMES` frozenset + hardcoded dispatch
(P2-14): it is the single place that knows which tool names are terminal (finalizers)
and, per doc's Contract C3, which gates guard them (structural before sufficiency).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from src.schemas.agent_findings import AgentFindings, AnalyticalFindings
from src.services.chat.agent.gates import (
    GateFn,
    analytical_insufficiency_gate,
    confirmed_absent_gate,
    missing_entity_gate,
    named_item_gate,
    restatement_integrity_gate,
)
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


@dataclass(frozen=True)
class ToolRegistration:
    schema: dict
    terminal: bool = False
    gates: tuple[GateFn, ...] = field(default_factory=tuple)


TOOL_REGISTRY: dict[str, ToolRegistration] = {
    "search_documents": ToolRegistration(schema=SEARCH_TOOL),
    "report_findings": ToolRegistration(
        schema=REPORT_FINDINGS_TOOL, terminal=True, gates=(missing_entity_gate,)
    ),
    "report_analytical_findings": ToolRegistration(
        # Order matters — `loop.py` stops at the first rejection, so the most specific,
        # most damaging complaint must come first (FR-13, D8):
        #   1. restatement_integrity — content already established is being lost; fixing
        #      anything else on top of a degraded restatement bakes the loss in.
        #   2. confirmed_absent — an unbacked absence claim is a fabricated finding.
        #   3. named_item — a specific item still needs its follow-up search.
        #   4. analytical_insufficiency — the generic thinness complaint, last.
        schema=REPORT_ANALYTICAL_TOOL,
        terminal=True,
        gates=(
            restatement_integrity_gate,
            confirmed_absent_gate,
            named_item_gate,
            analytical_insufficiency_gate,
        ),
    ),
}

# Stage 1.5: one tool pool for every query_shape. Each gate is registered against the
# specific finalizer it guards (missing_entity_gate only fires for report_findings,
# analytical_insufficiency_gate only for report_analytical_findings), so handing the
# model both finalizers unconditionally does not change which gate fires for which
# candidate type — it only removes the branch that built two separate tool lists.
# Prompt selection (v3_agent vs v4_agent_analytical) still varies by query_shape.
ALL_TOOLS = [
    TOOL_REGISTRY["search_documents"].schema,
    TOOL_REGISTRY["report_findings"].schema,
    TOOL_REGISTRY["report_analytical_findings"].schema,
]


def is_terminal(name: str) -> bool:
    reg = TOOL_REGISTRY.get(name)
    return reg is not None and reg.terminal


def gates_for(name: str) -> tuple[GateFn, ...]:
    reg = TOOL_REGISTRY.get(name)
    return reg.gates if reg is not None else ()
