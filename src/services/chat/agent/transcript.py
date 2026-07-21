"""Transcript — the model's view of the conversation.

Token-bounded, ordered, lossy, compacted, written in prose. It is a *view*, not a
record — compaction is allowed to destroy it, because `EvidenceLedger` (evidence.py)
retains what it drops (Contract C1). No compaction-behavior change in this step; this
is the move of `_compress_history` plus deletion of its no-op branch (P2-12).
"""

from __future__ import annotations

import json
from dataclasses import replace

from src.services.llm_adapters.base_adapter import ChatMessage, Role, ToolCallRef


def assistant_msg_with_tool_calls(tool_calls: list[ToolCallRef]) -> ChatMessage:
    return ChatMessage(role=Role.assistant, content=None, tool_calls=tuple(tool_calls))


def stub_rejected_tool_call(tc: ToolCallRef) -> ToolCallRef:
    """Strip a rejected finalizer call's claim/evidence payload before it re-enters history.

    Otherwise the model keeps seeing its own rejected draft claims verbatim (assistant
    tool-call messages survive compaction), inviting it to copy a stale claim into the
    eventually-accepted call without re-deriving fresh evidence for it.
    """
    return replace(tc, arguments=json.dumps({"status": "rejected"}))


def _compress_history(messages: list[ChatMessage], keep_last_n_turns: int) -> list[ChatMessage]:
    """Replace tool results from turns older than keep_last_n_turns with compact stubs.

    A "turn" is an assistant message that contains tool_calls followed by its tool
    result messages. Whole turns are truncated so the agent never sees a partial view
    of a prior turn's evidence.
    """
    turn_starts: list[int] = [
        i for i, m in enumerate(messages) if m.role == Role.assistant and m.tool_calls
    ]
    if len(turn_starts) <= keep_last_n_turns:
        return messages

    cutoff_idx = turn_starts[-(keep_last_n_turns)]
    stale: set[int] = set()
    for i, m in enumerate(messages):
        if i >= cutoff_idx:
            break
        if m.role in (Role.tool, Role.assistant) and (m.role == Role.tool or m.tool_calls):
            stale.add(i)

    result = []
    for i, m in enumerate(messages):
        if i in stale and m.role == Role.tool:
            result.append(
                ChatMessage(
                    role=m.role,
                    tool_call_id=m.tool_call_id,
                    content="[turn results truncated — already incorporated]",
                )
            )
        else:
            # Stale assistant messages are kept as-is so tool_call_id references stay
            # valid; non-stale messages pass through unchanged either way.
            result.append(m)
    return result


class Transcript:
    def __init__(self, messages: list[ChatMessage]) -> None:
        self.messages = messages

    def append(self, message: ChatMessage) -> None:
        self.messages.append(message)

    def append_tool_calls(self, tool_calls: list[ToolCallRef]) -> None:
        self.messages.append(assistant_msg_with_tool_calls(tool_calls))

    def compress(self, keep_last_n_turns: int = 2) -> None:
        self.messages = _compress_history(self.messages, keep_last_n_turns)
