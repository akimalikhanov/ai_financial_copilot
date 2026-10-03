"""Transcript — the model's view of the run.

Append-only: no earlier message is ever rewritten. The provider caches the whole prefix,
and every S-label the model can cite stays on screen, so a claim can only be grounded on
text the model can still read. `EvidenceLedger` (evidence.py) remains the record that
resolves those labels.

The run is bounded by the USD budget and the iteration cap, not by evicting old results.
"""

from __future__ import annotations

from src.services.llm_adapters.base_adapter import ChatMessage, Role, ToolCallRef


def assistant_msg_with_tool_calls(tool_calls: list[ToolCallRef]) -> ChatMessage:
    return ChatMessage(role=Role.assistant, content=None, tool_calls=tuple(tool_calls))


class Transcript:
    def __init__(self, messages: list[ChatMessage]) -> None:
        self.messages = messages

    def append(self, message: ChatMessage) -> None:
        self.messages.append(message)

    def append_tool_calls(self, tool_calls: list[ToolCallRef]) -> None:
        self.messages.append(assistant_msg_with_tool_calls(tool_calls))
