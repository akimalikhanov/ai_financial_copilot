"""Transcript — the model's view of the conversation.

Token-bounded, ordered, lossy, compacted, written in prose. It is a *view*, not a
record — compaction is allowed to destroy it, because `EvidenceLedger` (evidence.py)
retains what it drops (Contract C1).

Compaction is evidence-aware and aggressive: it evicts the bulky rendered tool-result
context of all but the most recent turn while keeping the assistant's reasoning and
tool-call structure. That is safe precisely because the ledger owns chunk_id ↔ S-label
— an evicted tool result drops the label from the model's *view*, but the model can
still cite it and it resolves via `EvidenceLedger.resolve_refs` (Contract C1). No LLM
call; this is tool-result eviction (step 9, work item 1 — most of the win).

An evicted result is not blanked to a generic stub: it keeps a one-line breadcrumb of
which search produced it (entity + query) and which labels it yielded (e.g. S1–S8),
so the model still knows what it already has and can cite those labels without
re-searching.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from src.services.llm_adapters.base_adapter import ChatMessage, Role, ToolCallRef

if TYPE_CHECKING:
    from src.services.chat.agent.evidence import EvidenceLedger

# Excerpts are rendered as <retrieved_excerpt id="Sn" ...> (context_assembler); the
# label is the stable handle the ledger resolves, so an evicted result keeps its
# label range as a breadcrumb.
_LABEL_RE = re.compile(r'id="(S\d+)"')

# Named locally rather than imported from tools.py: this module deliberately depends on
# nothing in the agent package (it is a pure view over ChatMessage).
_REPORT_TOOL_NAMES = frozenset({"report_findings", "report_analytical_findings"})


def assistant_msg_with_tool_calls(tool_calls: list[ToolCallRef]) -> ChatMessage:
    return ChatMessage(role=Role.assistant, content=None, tool_calls=tuple(tool_calls))


def _summarize_evicted(content: str, call: ToolCallRef | None) -> str:
    """A compact, still-informative replacement for an evicted search result.

    The bulky excerpt bodies go; what the model needs to keep reasoning stays: which
    search this was (entity + query, from the surviving tool call) and which labels it
    yielded — those still resolve via the ledger, so the model can cite them without
    re-searching (Contract C1).
    """
    nums = sorted(int(m[1:]) for m in _LABEL_RE.findall(content))
    span = f"S{nums[0]}" if len(nums) == 1 else f"S{nums[0]}–S{nums[-1]}"
    body = f"{len(nums)} excerpt{'s' if len(nums) != 1 else ''} {span}, still citable by label"
    if call is not None:
        try:
            args = json.loads(call.arguments)
            query = str(args.get("query") or "")
            query = query[:80] + "…" if len(query) > 80 else query
            head = f'{call.name}(entity="{args.get("entity") or "?"}", query="{query}")'
        except (json.JSONDecodeError, TypeError):
            head = call.name
        return f"[compacted] {head} → {body}"
    return f"[compacted] {body}"


def cap_history(
    messages: list[ChatMessage],
    max_turns: int,
    max_assistant_chars: int,
) -> list[ChatMessage]:
    """Trim prior conversation to the last `max_turns` user/assistant pairs.

    Prior conversation was permanent and *token-unbounded* in the agent transcript — up
    to 50 messages carried on every turn, dwarfing the system prompt and rivalling the
    excerpts it exists to contextualize (10b §4, row 2). Assistant content is truncated
    hardest: it is the model's own prose, recoverable-in-gist, while a prior user turn is
    the only record of what was asked.
    """
    if max_turns <= 0:
        return []
    # A "turn" starts at its user message, so the window starts at the Nth-from-last user
    # message — never mid-pair, which would carry an answer whose question was dropped.
    user_idx = [i for i, m in enumerate(messages) if m.role == Role.user]
    start = user_idx[-max_turns] if len(user_idx) > max_turns else 0
    kept: list[ChatMessage] = []
    for m in messages[start:]:
        if m.role == Role.assistant and m.content and len(m.content) > max_assistant_chars:
            m = ChatMessage(  # noqa: PLW2901 — truncated copy, the original is untouched
                role=m.role,
                content=m.content[:max_assistant_chars] + "…",
                tool_calls=m.tool_calls,
                tool_call_id=m.tool_call_id,
            )
        kept.append(m)
    return kept


def _compress_history(
    messages: list[ChatMessage], keep_last_n_turns: int
) -> tuple[list[ChatMessage], list[str]]:
    """Replace rendered search results from turns older than keep_last_n_turns with a
    compact summary stub (which search, which labels), keeping the turn structure.

    Returns (messages, evicted_labels). The labels go to `EvidenceLedger.mark_evicted` so
    the ledger knows what left the model's view; this module stays ledger-free (single
    writer, Contract C2).

    A "turn" is an assistant message that contains tool_calls followed by its tool
    result messages. Whole turns are compacted so the agent never sees a partial view
    of a prior turn's evidence. Only tool results carrying rendered excerpts (an S-label)
    are compacted — error/rejection notices have no labels and pass through untouched, so
    the model never loses *why* something was rejected.
    """
    # A turn boundary is an assistant message that issued at least one *search*. Counting
    # report-only turns here would shift the cutoff and evict real excerpts a turn early —
    # live now that reports are non-terminal and can arrive in their own turn (10b §4b).
    turn_starts: list[int] = [
        i
        for i, m in enumerate(messages)
        if m.role == Role.assistant
        and m.tool_calls
        and any(tc.name not in _REPORT_TOOL_NAMES for tc in m.tool_calls)
    ]
    if len(turn_starts) <= keep_last_n_turns:
        return messages, []

    cutoff_idx = turn_starts[-(keep_last_n_turns)]
    calls_by_id: dict[str, ToolCallRef] = {
        tc.id: tc for m in messages for tc in (m.tool_calls or ())
    }

    result = []
    evicted: list[str] = []
    for i, m in enumerate(messages):
        if i < cutoff_idx and m.role == Role.tool and _LABEL_RE.search(m.content or ""):
            evicted.extend(_LABEL_RE.findall(m.content or ""))
            result.append(
                ChatMessage(
                    role=Role.tool,
                    tool_call_id=m.tool_call_id,
                    content=_summarize_evicted(
                        m.content or "", calls_by_id.get(m.tool_call_id or "")
                    ),
                )
            )
        else:
            # Assistant tool-call messages are kept as-is so their tool_calls (entity +
            # query) — and the tool_call_id linkage — survive; everything else passes
            # through unchanged.
            result.append(m)
    return result, evicted


class Transcript:
    def __init__(self, messages: list[ChatMessage]) -> None:
        self.messages = messages

    def append(self, message: ChatMessage) -> None:
        self.messages.append(message)

    def append_tool_calls(self, tool_calls: list[ToolCallRef]) -> None:
        self.messages.append(assistant_msg_with_tool_calls(tool_calls))

    def compress(self, evidence: EvidenceLedger | None = None, keep_last_n_turns: int = 2) -> None:
        """Evict bulky tool-result context from all but the most recent turn(s).

        Aggressive by default (`keep_last_n_turns=1`): only the latest turn's rendered
        chunks stay in the model's view. Licensed by Contract C1 — every evicted S-label
        still resolves through `EvidenceLedger`, so the model can cite it regardless.

        The ledger is told which labels left the view so a later search that re-returns
        one re-renders it instead of assuming the model can still read it.
        """
        self.messages, evicted = _compress_history(self.messages, keep_last_n_turns)
        if evidence is not None and evicted:
            evidence.mark_evicted(evicted)
