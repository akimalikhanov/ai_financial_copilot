"""Lightweight coverage for the agent-specific SSE event builders in events.py.

test_chat_full_flow.py already covers error_event/build_usage_event/
extract_used_citations/build_references_list end-to-end via the classic
(non-agent) pipeline; this file only adds the agent-loop event shapes that
aren't exercised there.
"""

from __future__ import annotations

from src.services.chat.events import (
    agent_synthesis_starting_event,
    agent_turn_started_event,
    tool_call_completed_event,
    tool_call_started_event,
)


def test_agent_turn_started_event() -> None:
    assert agent_turn_started_event(2) == {"iteration": 2}


def test_tool_call_started_event() -> None:
    assert tool_call_started_event("Acme", "hybrid") == {
        "entity": "Acme",
        "search_mode": "hybrid",
    }


def test_tool_call_completed_event() -> None:
    assert tool_call_completed_event("Acme", chunks_returned=5, new_chunks_added=3) == {
        "entity": "Acme",
        "chunks_returned": 5,
        "new_chunks_added": 3,
    }


def test_agent_synthesis_starting_event() -> None:
    assert agent_synthesis_starting_event(total_chunks=10, iterations=2) == {
        "total_chunks": 10,
        "iterations": 2,
    }
