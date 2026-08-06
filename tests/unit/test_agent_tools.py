"""Unit tests for generated agent tool schemas.

The Pydantic arg models are the single source of truth: their JSON schemas drive
the tool definitions handed to the LLM, and the same models parse the arguments
back. These tests prove schema and parser cannot drift — a payload constructed to
match a generated schema validates against the model that generated it.
"""

from __future__ import annotations

import json

from src.schemas.agent_findings import AgentFindings, AnalyticalFindings
from src.services.chat.agent.tools import (
    ANALYTICAL_TOOLS,
    EXTRACTION_TOOLS,
    REPORT_ANALYTICAL_TOOL,
    REPORT_FINDINGS_TOOL,
    REPORT_TOOL_NAMES,
    SEARCH_ANALYTICAL_TOOL,
    SEARCH_TOOL,
    SearchDocumentsArgs,
    tool_schema,
)


def _params(tool: dict) -> dict:
    return tool["function"]["parameters"]


class TestSchemaShape:
    def test_search_tool_names_and_params(self) -> None:
        assert SEARCH_TOOL["function"]["name"] == "search_documents"
        params = _params(SEARCH_TOOL)
        # Extraction keeps the two-field search: make_strict forces every property into
        # `required`, so offering sub_question here would oblige a null for a concept
        # v3_agent never explains.
        assert set(params["properties"]) == {"entity", "query"}
        assert set(params["required"]) == {"entity", "query"}

    def test_analytical_search_tool_carries_sub_question(self) -> None:
        assert SEARCH_ANALYTICAL_TOOL["function"]["name"] == "search_documents"
        params = _params(SEARCH_ANALYTICAL_TOOL)
        assert set(params["properties"]) == {"entity", "query", "sub_question"}
        # Required, so the plan seeds from every analytical search rather than whichever
        # ones the model remembered to decompose.
        assert set(params["required"]) == {"entity", "query", "sub_question"}

    def test_report_findings_tool_name(self) -> None:
        assert REPORT_FINDINGS_TOOL["function"]["name"] == "report_findings"
        assert "findings" in _params(REPORT_FINDINGS_TOOL)["properties"]

    def test_report_analytical_tool_name(self) -> None:
        assert REPORT_ANALYTICAL_TOOL["function"]["name"] == "report_analytical_findings"
        assert "observations" in _params(REPORT_ANALYTICAL_TOOL)["properties"]

    def test_make_strict_applied(self) -> None:
        # Every object node is additionalProperties:false with an exhaustive required list.
        params = _params(REPORT_FINDINGS_TOOL)
        assert params["additionalProperties"] is False
        assert set(params["required"]) == set(params["properties"].keys())


class TestRoundTrip:
    """Generate schema -> build a matching tool-call payload -> parse it back."""

    def test_search_args_round_trip(self) -> None:
        payload = json.dumps({"entity": "Acme Corp", "query": "revenue 2023"})
        args = SearchDocumentsArgs.model_validate_json(payload)
        assert args.entity == "Acme Corp"
        assert args.query == "revenue 2023"

    def test_report_findings_round_trip(self) -> None:
        payload = json.dumps(
            {
                "metric_requested": "revenue",
                "comparison_op": "argmax",
                "findings": [
                    {
                        "entity": "Acme",
                        "available": True,
                        "value": 1234.5,
                        "currency": "USD",
                        "period_end": "2023-12-31",
                        "source_chunks": ["S1", "S3"],
                        "reason": None,
                        "unit": "M",
                    }
                ],
            }
        )
        parsed = AgentFindings.model_validate(json.loads(payload))
        assert parsed.metric_requested == "revenue"
        assert parsed.comparison_op == "argmax"
        assert len(parsed.findings) == 1
        assert parsed.findings[0].source_chunks == ["S1", "S3"]

    def test_report_findings_minimal_defaults(self) -> None:
        # source_chunks omitted -> defaults to [] (schema marks it non-nullable).
        parsed = AgentFindings.model_validate(
            {"metric_requested": "revenue", "findings": [{"entity": "Acme", "available": False}]}
        )
        assert parsed.findings[0].source_chunks == []
        assert parsed.comparison_op is None

    def test_report_analytical_round_trip(self) -> None:
        payload = json.dumps(
            {
                "question": "Why did margins fall?",
                "conclusion": "Input costs rose.",
                "gaps": None,
                "observations": [
                    {
                        "aspect": "cogs",
                        "claim": "COGS rose 12%",
                        "evidence_chunks": ["S2"],
                        "confidence": "high",
                        "refuted_by": None,
                    }
                ],
            }
        )
        parsed = AnalyticalFindings.model_validate(json.loads(payload))
        assert parsed.question == "Why did margins fall?"
        assert len(parsed.observations) == 1
        assert parsed.observations[0].confidence == "high"


class TestFieldDescriptionsPreserved:
    def test_source_chunks_description_carried_into_schema(self) -> None:
        defs = _params(REPORT_FINDINGS_TOOL)["$defs"]["EntityFinding"]["properties"]
        assert "Excerpt IDs" in defs["source_chunks"]["description"]

    def test_tool_schema_helper_wraps_model(self) -> None:
        schema = tool_schema("x", "does x", SearchDocumentsArgs)
        assert schema["type"] == "function"
        assert schema["function"]["description"] == "does x"


class TestPerPathToolPools:
    """Step 7: each path names its own pool, and no tool is terminal."""

    def test_analytical_pool_offers_only_its_own_finalizer(self) -> None:
        names = [t["function"]["name"] for t in ANALYTICAL_TOOLS]
        assert names == ["search_documents", "report_analytical_findings"]

    def test_extraction_pool_offers_only_its_own_finalizer(self) -> None:
        # report_analytical_findings leaving extraction's view is the point: v3_agent
        # never named it, and an Observation on an extraction run flips the ledger kind.
        names = [t["function"]["name"] for t in EXTRACTION_TOOLS]
        assert names == ["search_documents", "report_findings"]

    def test_neither_pool_sees_the_other_paths_finalizer(self) -> None:
        analytical = {t["function"]["name"] for t in ANALYTICAL_TOOLS}
        extraction = {t["function"]["name"] for t in EXTRACTION_TOOLS}
        assert "report_findings" not in analytical
        assert "report_analytical_findings" not in extraction

    def test_report_tool_names_spans_both_pools(self) -> None:
        # The loop partitions each turn on this set, so a report tool missing from it
        # would be routed to the search path and executed as a search.
        pooled = {t["function"]["name"] for t in (*ANALYTICAL_TOOLS, *EXTRACTION_TOOLS)}
        assert REPORT_TOOL_NAMES.issubset(pooled)
        assert {"report_findings", "report_analytical_findings"} == REPORT_TOOL_NAMES
        assert "search_documents" not in REPORT_TOOL_NAMES

    def test_no_terminal_or_gate_machinery_remains(self) -> None:
        # D3 deleted the registry: nothing ends the run but the loop's own coverage check,
        # so a re-introduced `terminal` flag would silently restore the one-shot finalizer.
        import src.services.chat.agent.tools as tools_module

        for attr in ("TOOL_REGISTRY", "is_terminal", "gates_for", "ToolRegistration"):
            assert not hasattr(tools_module, attr), f"{attr} should have been deleted by D3"

    def test_report_descriptions_invite_incremental_calls(self) -> None:
        # The schema description is the only place the model is told it may report more
        # than once; "call this ONCE ... ends the search phase" is what it replaced.
        for tool in (REPORT_FINDINGS_TOOL, REPORT_ANALYTICAL_TOOL):
            desc = tool["function"]["description"].lower()
            assert "more than once" in desc
            assert "ends the search phase" not in desc
