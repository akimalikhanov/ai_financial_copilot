"""Unit tests for generated agent tool schemas.

The Pydantic arg models are the single source of truth: their JSON schemas drive
the tool definitions handed to the LLM, and the same models parse the arguments
back. These tests prove schema and parser cannot drift — a payload constructed to
match a generated schema validates against the model that generated it.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from src.schemas.agent_findings import FindingsReport
from src.services.chat.agent.state import AgentSettings, shape_config
from src.services.chat.agent.tools import (
    REPORT_FINDINGS_TOOL,
    REPORT_TOOL_NAME,
    SEARCH_ANALYTICAL_TOOL,
    SEARCH_TOOL,
    SearchDocumentsArgs,
    tool_schema,
)


def _params(tool: dict) -> dict:
    return tool["function"]["parameters"]


def _settings() -> AgentSettings:
    return AgentSettings(
        tool_model="m",
        max_iterations=5,
        cost_budget_usd=0.1,
        max_concurrent_searches=3,
        max_chunks_per_entity=5,
        max_empty_rounds=1,
        turn_timeout_seconds=60,
        deadline_seconds=180,
        max_iterations_analytical=7,
        max_plan_items=6,
    )


class TestSchemaShape:
    def test_search_tool_names_and_params(self) -> None:
        assert SEARCH_TOOL["function"]["name"] == "search_documents"
        params = _params(SEARCH_TOOL)
        # No sub_question on extraction: make_strict forces every property into
        # `required`, so offering it here would oblige a null for a concept its prompt
        # never explains.
        assert set(params["properties"]) == {"entity", "query", "keywords"}
        assert set(params["required"]) == {"entity", "query", "keywords"}

    def test_analytical_search_tool_carries_sub_question(self) -> None:
        assert SEARCH_ANALYTICAL_TOOL["function"]["name"] == "search_documents"
        params = _params(SEARCH_ANALYTICAL_TOOL)
        assert set(params["properties"]) == {"entity", "query", "keywords", "sub_question"}
        # Required, so the plan seeds from every analytical search rather than whichever
        # ones the model remembered to decompose.
        assert set(params["required"]) == {"entity", "query", "keywords", "sub_question"}

    def test_keywords_is_a_non_nullable_string_in_both_schemas(self) -> None:
        # Nullable would let the model skip it; the BM25 query then falls back to `query`.
        for tool in (SEARCH_TOOL, SEARCH_ANALYTICAL_TOOL):
            assert _params(tool)["properties"]["keywords"]["type"] == "string"

    def test_parser_accepts_a_call_without_keywords(self) -> None:
        args = SearchDocumentsArgs.model_validate_json('{"entity": "Acme", "query": "revenue"}')
        assert args.keywords is None

    def test_report_findings_tool_name(self) -> None:
        assert REPORT_FINDINGS_TOOL["function"]["name"] == REPORT_TOOL_NAME == "report_findings"
        props = _params(REPORT_FINDINGS_TOOL)["properties"]
        assert set(props) == {"findings", "comparison_op", "conclusion"}

    def test_make_strict_applied(self) -> None:
        # Every object node is additionalProperties:false with an exhaustive required list.
        params = _params(REPORT_FINDINGS_TOOL)
        assert params["additionalProperties"] is False
        assert set(params["required"]) == set(params["properties"].keys())
        for name in ("Finding", "Figure"):
            node = params["$defs"][name]
            assert node["additionalProperties"] is False
            assert set(node["required"]) == set(node["properties"])

    def test_served_fields_are_not_advertised(self) -> None:
        # `unresolved` is loop-written: advertising it would let the model author lines
        # that close nothing.
        assert "unresolved" not in _params(REPORT_FINDINGS_TOOL)["properties"]


class TestRoundTrip:
    """Generate schema -> build a matching tool-call payload -> parse it back."""

    def test_search_args_round_trip(self) -> None:
        payload = json.dumps({"entity": "Acme Corp", "query": "revenue 2023"})
        args = SearchDocumentsArgs.model_validate_json(payload)
        assert args.entity == "Acme Corp"
        assert args.query == "revenue 2023"

    def test_report_with_figures_round_trip(self) -> None:
        payload = json.dumps(
            {
                "comparison_op": "argmax",
                "conclusion": None,
                "findings": [
                    {
                        "key": "Acme",
                        "claim": "Acme's revenue rose.",
                        "supported": True,
                        "evidence": ["S1", "S3"],
                        "confidence": "high",
                        "figures": [
                            {
                                "metric": "revenue",
                                "amount": 1234.5,
                                "unit": "M",
                                "currency": "USD",
                                "period_end": "2023-12-31",
                                "fiscal_label": "FY2023",
                            },
                            {
                                "metric": "revenue",
                                "amount": 1100.0,
                                "unit": "M",
                                "currency": "USD",
                                "period_end": "2022-12-31",
                                "fiscal_label": "FY2022",
                            },
                        ],
                    }
                ],
            }
        )
        parsed = FindingsReport.model_validate_json(payload)
        assert parsed.comparison_op == "argmax"
        assert [f.period_end for f in parsed.findings[0].figures] == ["2023-12-31", "2022-12-31"]
        assert parsed.findings[0].evidence == ["S1", "S3"]

    def test_non_numeric_finding_needs_no_figures(self) -> None:
        parsed = FindingsReport.model_validate(
            {
                "findings": [
                    {
                        "key": "A1",
                        "claim": "Acme introduced a special dividend.",
                        "supported": True,
                        "evidence": ["S2"],
                        "confidence": "medium",
                    }
                ]
            }
        )
        assert parsed.findings[0].figures == []
        assert parsed.comparison_op is None

    def test_unknown_unit_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FindingsReport.model_validate(
                {
                    "findings": [
                        {
                            "key": "Acme",
                            "claim": "c",
                            "supported": True,
                            "evidence": [],
                            "confidence": "high",
                            "figures": [
                                {
                                    "metric": "revenue",
                                    "amount": 1.0,
                                    "unit": "millions",
                                    "currency": None,
                                    "period_end": None,
                                    "fiscal_label": None,
                                }
                            ],
                        }
                    ]
                }
            )


class TestFieldDescriptionsPreserved:
    def test_evidence_description_carried_into_schema(self) -> None:
        props = _params(REPORT_FINDINGS_TOOL)["$defs"]["Finding"]["properties"]
        assert "Excerpt IDs" in props["evidence"]["description"]

    def test_tool_schema_helper_wraps_model(self) -> None:
        schema = tool_schema("x", "does x", SearchDocumentsArgs)
        assert schema["type"] == "function"
        assert schema["function"]["description"] == "does x"


class TestShapePools:
    """Every shape reports through one tool; only the search schema differs."""

    def test_analytical_pool(self) -> None:
        assert shape_config("analytical", _settings()).tools == [
            SEARCH_ANALYTICAL_TOOL,
            REPORT_FINDINGS_TOOL,
        ]

    @pytest.mark.parametrize("shape", ["extraction", "comparison", None])
    def test_extraction_pool(self, shape: str | None) -> None:
        assert shape_config(shape, _settings()).tools == [SEARCH_TOOL, REPORT_FINDINGS_TOOL]

    def test_no_terminal_or_gate_machinery_remains(self) -> None:
        # Nothing ends the run but the loop's own coverage check, so a re-introduced
        # `terminal` flag would silently restore the one-shot finalizer.
        import src.services.chat.agent.tools as tools_module

        for attr in ("TOOL_REGISTRY", "is_terminal", "gates_for", "ToolRegistration"):
            assert not hasattr(tools_module, attr)

    def test_report_description_invites_incremental_calls(self) -> None:
        # The schema description is the only place the model is told it may report more
        # than once.
        desc = REPORT_FINDINGS_TOOL["function"]["description"].lower()
        assert "more than once" in desc
