"""Transcript — the model's view of the run.

Append-only: no earlier message is ever rewritten. The provider caches the whole prefix,
and every S-label the model can cite stays on screen, so a claim can only be grounded on
text the model can still read. `EvidenceLedger` (evidence.py) remains the record that
resolves those labels.

The run is bounded by the USD budget and the iteration cap, not by evicting old results.
"""

from __future__ import annotations

from dataclasses import replace

from src.services.llm_adapters.base_adapter import ChatMessage, Role, ToolCallRef


def assistant_msg_with_tool_calls(tool_calls: list[ToolCallRef]) -> ChatMessage:
    return ChatMessage(role=Role.assistant, content=None, tool_calls=tuple(tool_calls))


def with_status(messages: list[ChatMessage], status: str | None) -> list[ChatMessage]:
    """The request view: `status` appended to the last tool result, never stored.

    A trailing user message would end the provider's turn, and `current_turn` reasoning
    models drop every reasoning item before it. Appended rather than prepended, so the
    next call's cached prefix still covers that result's content. Call 0 has no tool
    result yet, and no reasoning to lose, so the status goes as a user message there.
    """
    if not status:
        return list(messages)
    block = f"<run_status>\n{status}\n</run_status>"
    if messages and messages[-1].role is Role.tool:
        last = messages[-1]
        return [*messages[:-1], replace(last, content=f"{last.content or ''}\n\n{block}")]
    return [*messages, ChatMessage(role=Role.user, content=block)]


class Transcript:
    def __init__(self, messages: list[ChatMessage]) -> None:
        self.messages = messages

    def append(self, message: ChatMessage) -> None:
        self.messages.append(message)

    def append_tool_calls(self, tool_calls: list[ToolCallRef]) -> None:
        self.messages.append(assistant_msg_with_tool_calls(tool_calls))
