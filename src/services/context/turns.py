"""Prior conversation, built once per request and capped per model.

`prior_turns` is the only path from the chat tail to any model's history. The rules every
consumer needs live here, so no caller can skip one:

- messages are paired into turns first, so every window starts at a question;
- the current question is split off;
- every prior question is scanned, and a blocked turn is dropped together with its answer;
- history reads `content`, never `raw_content`. Only `raw_content` keeps a prior run's
  `[Sn]` markers, which would collide with this run's labels;
- `Turn` has no field for a findings block (Contract F1).

Each consumer then caps the turns with its own `HistoryBudget`, read from config on every
request, and formats them itself.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from src.schemas.chat import ChatMessage, Role, Turn
from src.services.llm_adapters.base_adapter import ChatMessage as AdapterChatMessage
from src.services.llm_adapters.base_adapter import Role as AdapterRole
from src.services.security.injection_detector import scan_user_input
from src.utils.config import (
    get_agent_history_budget_tokens,
    get_agent_history_max_answer_tokens,
    get_agent_history_step,
    get_answer_history_budget_tokens,
    get_answer_history_max_answer_tokens,
    get_answer_history_step,
    get_router_history_budget_tokens,
    get_router_history_max_answer_tokens,
    get_router_history_step,
    get_router_session_index_question_chars,
)

logger = logging.getLogger(__name__)

TRUNCATION_MARKER = " […truncated]"


@dataclass(frozen=True)
class HistoryBudget:
    budget_tokens: int
    # Per-answer cap. None keeps answers whole: a turn is either kept or dropped.
    max_answer_tokens: int | None = None
    # The window may start only at a turn index that is a multiple of `step`, so it moves
    # once every `step` turns and the cached prompt prefix holds in between.
    step: int = 1

    def __post_init__(self) -> None:
        if self.budget_tokens < 1 or self.step < 1:
            raise ValueError(f"invalid history budget: {self}")

    @classmethod
    def from_config(cls, budget_tokens: int, max_answer_tokens: int, step: int) -> HistoryBudget:
        """A configured budget; `max_answer_tokens` 0 means whole answers."""
        return cls(budget_tokens, max_answer_tokens or None, step)


def router_history() -> HistoryBudget:
    return HistoryBudget.from_config(
        get_router_history_budget_tokens(),
        get_router_history_max_answer_tokens(),
        get_router_history_step(),
    )


def tool_model_history() -> HistoryBudget:
    return HistoryBudget.from_config(
        get_agent_history_budget_tokens(),
        get_agent_history_max_answer_tokens(),
        get_agent_history_step(),
    )


def answer_history() -> HistoryBudget:
    return HistoryBudget.from_config(
        get_answer_history_budget_tokens(),
        get_answer_history_max_answer_tokens(),
        get_answer_history_step(),
    )


def approx_tokens(text: str) -> int:
    return -(-len(text) // 4)


def truncate_tokens(text: str, max_tokens: int) -> str:
    """`text` cut to about `max_tokens`, marked so a model doesn't read a cut-off table as
    complete."""
    if approx_tokens(text) <= max_tokens:
        return text
    return text[: max_tokens * 4].rstrip() + TRUNCATION_MARKER


def prior_turns(messages: Sequence[ChatMessage], *, scan: bool, request_id: str = "") -> list[Turn]:
    """Prior question/answer pairs, oldest first, without the current question."""
    msgs = list(messages)
    if msgs and msgs[-1].role == Role.user:
        msgs = msgs[:-1]

    pairs: list[tuple[ChatMessage, ChatMessage | None]] = []
    for m in msgs:
        if m.role == Role.user:
            pairs.append((m, None))
        elif m.role == Role.assistant and pairs and pairs[-1][1] is None:
            pairs[-1] = (pairs[-1][0], m)
        # An answer with no question before it (the tail was cut mid-pair) is dropped.

    turns: list[Turn] = []
    for index, (question, answer) in enumerate(pairs):
        text = question.content
        if scan and text:
            signal = scan_user_input(text)
            if signal.severity == "block":
                logger.info(
                    "history_turn_blocked",
                    extra={"request_id": request_id, "matched_rules": signal.matched_rules},
                )
                continue
            text = signal.sanitized_text
        turns.append(
            Turn(
                index=index,
                question=text,
                answer=(answer.content or None) if answer is not None else None,
                from_carryover=answer is not None and answer.answer_derived_from_carryover,
                summary=answer.turn_summary if answer is not None else None,
            )
        )
    return turns


def _turn_tokens(turn: Turn) -> int:
    return approx_tokens(turn.question) + approx_tokens(turn.answer or "")


def cap_turns(turns: Sequence[Turn], budget: HistoryBudget) -> list[Turn]:
    """The most recent whole turns that fit the budget, starting at a `step` boundary."""
    capped = [
        t.model_copy(update={"answer": truncate_tokens(t.answer, budget.max_answer_tokens)})
        if budget.max_answer_tokens is not None and t.answer is not None
        else t
        for t in turns
    ]
    if not capped:
        return []

    total = 0
    fits_from = len(capped)
    for pos in range(len(capped) - 1, -1, -1):
        total += _turn_tokens(capped[pos])
        if total > budget.budget_tokens:
            break
        fits_from = pos

    if fits_from == len(capped):
        # Not even the last turn fits: keep it, with its answer cut to what's left.
        last = capped[-1]
        room = max(budget.budget_tokens - approx_tokens(last.question), 0)
        if last.answer is None:
            return [last]
        return [last.model_copy(update={"answer": truncate_tokens(last.answer, room)})]

    first = capped[fits_from].index
    start = -(-first // budget.step) * budget.step
    window = [t for t in capped[fits_from:] if t.index >= start]
    return window or capped[fits_from:]


def as_messages(turns: Sequence[Turn]) -> list[AdapterChatMessage]:
    out: list[AdapterChatMessage] = []
    for t in turns:
        out.append(AdapterChatMessage(role=AdapterRole.user, content=t.question))
        if t.answer is not None:
            out.append(AdapterChatMessage(role=AdapterRole.assistant, content=t.answer))
    return out


def session_index(turns: Sequence[Turn]) -> str:
    """One line per prior turn: what was asked, about which entities, over how many
    documents. Built from stored router output and the user's own question; no
    model-written prose."""
    max_chars = get_router_session_index_question_chars()
    lines: list[str] = []
    for t in turns:
        question = t.question.replace("\n", " ")
        if len(question) > max_chars:
            question = question[:max_chars].rstrip() + "…"
        parts = [f"T{t.index + 1}"]
        s = t.summary
        if s is not None:
            parts.append(f"{s.route}/{s.query_shape}" if s.query_shape else s.route)
            if s.entities:
                parts.append(", ".join(s.entities))
            if s.route == "retrieval":
                parts.append(f"docs: {s.doc_count if s.doc_count is not None else 'all'}")
        parts.append(f'"{question}"')
        lines.append("  ".join(parts))
    return "\n".join(lines)
