"""Transcript — the model's view of the run.

Append-only: no message a request has carried is ever rewritten, so each request starts
with the previous one verbatim. The provider caches the whole prefix,
and every S-label the model can cite stays on screen, so a claim can only be grounded on
text the model can still read. `EvidenceLedger` (evidence.py) remains the record that
resolves those labels.

The run is bounded by the USD budget and the iteration cap, not by evicting old results.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace

from src.services.llm_adapters.base_adapter import (
    AssistantTurnResult,
    ChatMessage,
    Role,
    ToolCallRef,
)


def assistant_msg_with_tool_calls(tool_calls: list[ToolCallRef]) -> ChatMessage:
    return ChatMessage(role=Role.assistant, content=None, tool_calls=tuple(tool_calls))


def opaque_size(messages: Sequence[ChatMessage]) -> tuple[int, int]:
    """Count and serialized bytes of the opaque items in `messages`. Sizes only: the
    payloads are never read here."""
    items = [o for m in messages for o in m.opaque or ()]
    return len(items), sum(len(json.dumps(o.payload)) for o in items)


def with_status(messages: list[ChatMessage], status: str | None) -> list[ChatMessage]:
    """`messages` with `status` appended to the last tool result.

    A trailing user message would end the provider's turn, and `current_turn` reasoning
    models drop every reasoning item before it. Call 0 has no tool result yet, and no
    reasoning to lose, so the status goes as a user message there.
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

    def add_status(self, status: str | None) -> None:
        """Store the status before the call that sends it. Responses only reuses a whole
        earlier prompt from cache, so a status sent but not stored makes every later call
        miss past call 0. The last tool result hasn't been sent yet, so folding the status
        into it rewrites nothing a request has carried."""
        self.messages[:] = with_status(self.messages, status)

    def append_turn(self, turn: AssistantTurnResult) -> None:
        """The turn's calls, with the provider items that produced them (`opaque`) for the
        adapter to replay. Prose stays out (`content=None`)."""
        self.messages.append(
            ChatMessage(
                role=Role.assistant,
                content=None,
                tool_calls=tuple(turn.tool_calls),
                opaque=turn.opaque or None,
            )
        )
