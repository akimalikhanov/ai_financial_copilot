"""Unit tests for generated agent tool schemas.

The Pydantic arg models are the single source of truth: their JSON schemas drive
the tool definitions handed to the LLM, and the same models parse the arguments
back. These tests prove schema and parser cannot drift — a payload constructed to
match a generated schema validates against the model that generated it.
"""

from __future__ import annotations

import json

from src.schemas.agent_findings import AgentFindings, AnalyticalFindings
from src.services.chat.agent.gates import missing_entity_gate
from src.services.chat.agent.tools import (
    ALL_TOOLS,
    REPORT_ANALYTICAL_TOOL,
    REPORT_FINDINGS_TOOL,
    SEARCH_TOOL,
    SearchDocumentsArgs,
    gates_for,
    is_terminal,
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

    def test_named_item_strict_shape(self) -> None:
        # Pins the model-facing contract for named_item: strict mode must require the key
        # (emitted as null when absent) and render it as a nullable $ref with default null,
        # exactly as refuted_by already renders. A make_strict change must not alter this.
        params = _params(REPORT_ANALYTICAL_TOOL)
        observation = params["$defs"]["Observation"]
        assert "named_item" in observation["required"]
        prop = observation["properties"]["named_item"]
        assert prop["default"] is None
        assert prop["anyOf"] == [{"$ref": "#/$defs/NamedItem"}, {"type": "null"}]

        named_item = params["$defs"]["NamedItem"]
        assert named_item["additionalProperties"] is False
        assert set(named_item["required"]) == {"name", "status"}

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
        # named_item absent from the payload still parses, defaulting to None.
        assert parsed.observations[0].named_item is None

    def test_report_analytical_named_item_round_trip(self) -> None:
        payload = json.dumps(
            {
                "question": "How did Payments do?",
                "observations": [
                    {
                        "aspect": "payments_revenue",
                        "claim": "Payments segment revenue is not broken out.",
                        "evidence_chunks": ["S1"],
                        "confidence": "medium",
                        "refuted_by": None,
                        "named_item": {"name": "Payments segment", "status": "unresolved"},
                    }
                ],
            }
        )
        parsed = AnalyticalFindings.model_validate(json.loads(payload))
        item = parsed.observations[0].named_item
        assert item is not None
        assert item.name == "Payments segment"
        assert item.status == "unresolved"


class TestFieldDescriptionsPreserved:
    def test_source_chunks_description_carried_into_schema(self) -> None:
        defs = _params(REPORT_FINDINGS_TOOL)["$defs"]["EntityFinding"]["properties"]
        assert "Excerpt IDs" in defs["source_chunks"]["description"]

    def test_tool_schema_helper_wraps_model(self) -> None:
        schema = tool_schema("x", "does x", SearchDocumentsArgs)
        assert schema["type"] == "function"
        assert schema["function"]["description"] == "does x"


class TestUnifiedToolPool:
    """Stage 1.5: one tool pool for every query_shape, not two hardcoded lists."""

    def test_all_tools_contains_both_finalizers(self) -> None:
        names = {t["function"]["name"] for t in ALL_TOOLS}
        assert names == {"search_documents", "report_findings", "report_analytical_findings"}

    def test_both_finalizers_are_terminal(self) -> None:
        assert is_terminal("report_findings")
        assert is_terminal("report_analytical_findings")
        assert not is_terminal("search_documents")

    def test_gates_scoped_to_their_own_finalizer(self) -> None:
        # missing_entity_gate only guards report_findings; analytical_insufficiency_gate
        # only guards report_analytical_findings — unifying the pool must not cross-wire
        # a gate onto the wrong finalizer.
        report_findings_gates = {g.__name__ for g in gates_for("report_findings")}
        report_analytical_gates = {g.__name__ for g in gates_for("report_analytical_findings")}
        assert report_findings_gates == {"missing_entity_gate"}
        assert report_analytical_gates == {
            "restatement_integrity_gate",
            "confirmed_absent_gate",
            "named_item_gate",
            "analytical_insufficiency_gate",
        }

    def test_analytical_gates_run_most_specific_first(self) -> None:
        # AC-14/FR-13: loop.py stops at the first rejection, so this ordering *is* the
        # implementation of "the most specific, most damaging complaint wins". Losing
        # established content outranks an unbacked absence claim, which outranks a
        # pending item, which outranks the generic thinness complaint.
        assert [g.__name__ for g in gates_for("report_analytical_findings")] == [
            "restatement_integrity_gate",
            "confirmed_absent_gate",
            "named_item_gate",
            "analytical_insufficiency_gate",
        ]

    def test_report_findings_gates_unchanged(self) -> None:
        # AC-9, FR-11: the named-item gate must not be cross-wired onto the other finalizer.
        assert gates_for("report_findings") == (missing_entity_gate,)
