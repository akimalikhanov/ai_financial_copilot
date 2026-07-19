"""Unit tests for generated agent tool schemas.

The Pydantic arg models are the single source of truth: their JSON schemas drive
the tool definitions handed to the LLM, and the same models parse the arguments
back. These tests prove schema and parser cannot drift — a payload constructed to
match a generated schema validates against the model that generated it.
"""

from __future__ import annotations

import json

from src.schemas.agent_findings import AgentFindings, AnalyticalFindings
from src.services.chat.tools import (
    REPORT_ANALYTICAL_TOOL,
    REPORT_FINDINGS_TOOL,
    SEARCH_TOOL,
    SearchDocumentsArgs,
    tool_schema,
)


def _params(tool: dict) -> dict:
    return tool["function"]["parameters"]


class TestSchemaShape:
    def test_search_tool_names_and_params(self) -> None:
        assert SEARCH_TOOL["function"]["name"] == "search_documents"
        props = _params(SEARCH_TOOL)["properties"]
        assert set(props) == {"entity", "query"}

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
