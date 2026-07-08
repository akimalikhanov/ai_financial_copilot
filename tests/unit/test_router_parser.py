"""Unit tests for parse_router_response (pure function, no mocking)."""

from __future__ import annotations

import json

from src.services.router.parser import parse_router_response


class TestParseRouterResponse:
    def test_empty_string_returns_empty_response_error(self) -> None:
        result, error = parse_router_response("")
        assert result is None
        assert error == "Empty response"

    def test_whitespace_only_returns_empty_response_error(self) -> None:
        result, error = parse_router_response("   \n\t  ")
        assert result is None
        assert error == "Empty response"

    def test_invalid_json_returns_error(self) -> None:
        result, error = parse_router_response("{not valid json")
        assert result is None
        assert error is not None
        assert error.startswith("Invalid JSON: ")

    def test_schema_validation_failure_returns_error(self) -> None:
        # missing required fields (route, user_intent, reasoning)
        result, error = parse_router_response(json.dumps({"foo": "bar"}))
        assert result is None
        assert error is not None
        assert error.startswith("Schema validation failed: ")

    def test_valid_response_parses_successfully(self) -> None:
        payload = {
            "route": "retrieval",
            "entities": [],
            "user_intent": "asking about revenue",
            "reasoning": "user wants a specific figure",
        }
        result, error = parse_router_response(json.dumps(payload))
        assert error is None
        assert result is not None
        assert result.route == "retrieval"
        assert result.user_intent == "asking about revenue"

    def test_reasoning_truncated_to_1000_chars(self) -> None:
        payload = {
            "route": "retrieval",
            "entities": [],
            "user_intent": "x",
            "reasoning": "a" * 2000,
        }
        result, _ = parse_router_response(json.dumps(payload))
        assert result is not None
        assert len(result.reasoning) == 1000

    def test_user_intent_truncated_to_300_chars(self) -> None:
        payload = {
            "route": "retrieval",
            "entities": [],
            "user_intent": "b" * 500,
            "reasoning": "x",
        }
        result, _ = parse_router_response(json.dumps(payload))
        assert result is not None
        assert len(result.user_intent) == 300
