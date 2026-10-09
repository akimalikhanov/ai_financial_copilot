"""Contract F1: carried findings must never reach any model's conversation history.

The tool model sees prior questions and synthesis prose. It must not see a prior turn's
findings block, nor prose derived from one — such a number has no plan entry, no
AspectStats, and no ledger backing it. The router and the answering model get the block
only through their own explicit channels, never through history.
"""

from __future__ import annotations

import dataclasses

import pytest

from src.models.message import Message, MessageRole
from src.schemas import chat as schemas
from src.schemas.query_router import RouterInput
from src.services.chat.agent.loop import CARRYOVER_STUB, agent_history
from src.services.context.conversation_history import _db_message_to_chat_message
from src.services.context.prompt_assembler import assemble_prompt
from src.services.context.turns import prior_turns
from src.services.llm_adapters.base_adapter import ChatMessage as AdapterChatMessage
from src.services.router.router import _build_messages

SENTINEL = "SENTINEL_FINDINGS_BLOCK_LEAK_CANARY"

BLOCK = f"""[STRUCTURED FINDINGS]
Metric: net income | Operation: argmin
Acme Corp             | USD -97,157.0K | native | period: 2022-12-31 | chunks: {SENTINEL}
[END STRUCTURED FINDINGS]"""

_CURRENT = schemas.ChatMessage(role=schemas.Role.user, content="and the year before?")


def _history() -> list[schemas.ChatMessage]:
    return [
        schemas.ChatMessage(role=schemas.Role.user, content="what was net income?"),
        schemas.ChatMessage(
            role=schemas.Role.assistant,
            content="Acme reported a net loss of USD 97,157K.",
            findings_block=BLOCK,
        ),
        _CURRENT,
    ]


def _carryover_history() -> list[schemas.ChatMessage]:
    return [
        schemas.ChatMessage(role=schemas.Role.user, content="summarize as a table"),
        schemas.ChatMessage(
            role=schemas.Role.assistant,
            content="| Metric | Value |\n| Net income | USD -97,157K |",
            answer_derived_from_carryover=True,
            findings_block=BLOCK,
        ),
        _CURRENT,
    ]


class _Renderer:
    def render_user_message(self, context: str, user_query: str) -> str:
        return f"{context}\n{user_query}"


def _answering_prompt(turns: list[schemas.Turn]) -> list[AdapterChatMessage]:
    return assemble_prompt(turns, "SYS", "", "q", renderer=_Renderer())  # type: ignore[arg-type]


def test_adapter_message_cannot_hold_a_findings_block() -> None:
    """The boundary is structural: the adapter type has no such field and is slotted."""
    fields = {f.name for f in dataclasses.fields(AdapterChatMessage)}
    assert "findings_block" not in fields
    assert getattr(AdapterChatMessage, "__dataclass_params__").frozen  # noqa: B009
    assert hasattr(AdapterChatMessage, "__slots__")

    # pyright flags this call itself — that is the contract holding at type-check time.
    with pytest.raises(TypeError):
        AdapterChatMessage(role=schemas.Role.user, content="x", findings_block=SENTINEL)  # type: ignore[call-arg]


def test_turn_cannot_hold_a_findings_block() -> None:
    assert not any("findings" in name for name in schemas.Turn.model_fields)
    assert SENTINEL not in repr(prior_turns(_history(), scan=False))


def test_findings_block_does_not_reach_agent_history() -> None:
    built = agent_history(prior_turns(_history(), scan=False))
    assert SENTINEL not in repr(built)
    # The assistant's own prose still crosses — that is intended.
    assert any("net loss of USD 97,157K" in (m.content or "") for m in built)


def test_findings_block_does_not_reach_router_or_answering_history() -> None:
    turns = prior_turns(_history(), scan=False)
    router_msgs = _build_messages(RouterInput(query="q", prior_turns=turns), system="SYS")
    assert SENTINEL not in repr(router_msgs)
    assert SENTINEL not in repr(_answering_prompt(turns))


def test_carryover_derived_prose_is_stubbed_for_the_tool_model() -> None:
    built = agent_history(prior_turns(_carryover_history(), scan=False))

    assert "97,157" not in repr(built)
    assert [m.content for m in built] == ["summarize as a table", CARRYOVER_STUB]


def test_carryover_derived_prose_reaches_the_answering_model() -> None:
    """The answering model wrote that table; it keeps its own prior answer."""
    answer_msgs = _answering_prompt(prior_turns(_carryover_history(), scan=False))
    assert any("97,157" in (m.content or "") for m in answer_msgs)


def test_normal_assistant_prose_is_not_stubbed() -> None:
    built = agent_history(prior_turns(_history(), scan=False))
    assert all(m.content != CARRYOVER_STUB for m in built)


def test_stub_holds_with_injection_scanning_on() -> None:
    """Scanning rewrites questions; the answer stub must be unaffected by it."""
    built = agent_history(prior_turns(_carryover_history(), scan=True))
    assert "97,157" not in repr(built)
    assert any(m.content == CARRYOVER_STUB for m in built)


def test_carryover_flag_and_summary_survive_the_cache_round_trip() -> None:
    """The stub and the session index only work if their fields come back off the tail."""
    summary = schemas.TurnSummary(route="retrieval", query_shape="extraction", entities=["Acme"])
    original = schemas.ChatMessage(
        role=schemas.Role.assistant,
        content="| Net income | USD -97,157K |",
        answer_derived_from_carryover=True,
        findings_block=BLOCK,
        findings_block_hops=2,
        turn_summary=summary,
    )
    restored = schemas.ChatMessage.model_validate(original.model_dump(mode="json"))
    assert restored.answer_derived_from_carryover is True
    assert restored.findings_block_hops == 2
    assert restored.turn_summary == summary

    question = schemas.ChatMessage(role=schemas.Role.user, content="table please")
    built = agent_history(prior_turns([question, restored], scan=False))
    assert SENTINEL not in repr(built)
    assert built[-1].content == CARRYOVER_STUB


def test_legacy_cache_entry_defaults_to_not_carried() -> None:
    """Older entries lack the newer keys; they must read as ordinary prose."""
    legacy = schemas.ChatMessage.model_validate({"role": "assistant", "content": "old answer"})
    assert legacy.answer_derived_from_carryover is False
    assert legacy.findings_block_hops == 0
    assert legacy.findings_block is None
    assert legacy.turn_summary is None

    question = schemas.ChatMessage(role=schemas.Role.user, content="q")
    assert agent_history(prior_turns([question, legacy], scan=False))[-1].content == "old answer"


def test_cold_cache_db_path_matches_the_cache_path() -> None:
    """On TTL expiry history is rebuilt from Postgres — it must carry the same fields,
    or the stub and the session index silently stop working once the cache drops."""
    row = Message(
        role=MessageRole.assistant,
        content="| Net income | USD -97,157K |",
        message_metadata={
            "findings_block": BLOCK,
            "answer_derived_from_carryover": True,
            "findings_block_hops": 2,
            "turn_summary": {"route": "direct_answer", "entities": ["Acme"]},
        },
    )
    msg = _db_message_to_chat_message(row)
    assert msg.answer_derived_from_carryover is True
    assert msg.findings_block_hops == 2
    assert msg.findings_block == BLOCK
    assert msg.turn_summary == schemas.TurnSummary(route="direct_answer", entities=["Acme"])

    question = schemas.ChatMessage(role=schemas.Role.user, content="q")
    built = agent_history(prior_turns([question, msg], scan=False))
    assert SENTINEL not in repr(built)
    assert built[-1].content == CARRYOVER_STUB


def test_cold_cache_legacy_row_defaults() -> None:
    row = Message(role=MessageRole.assistant, content="old answer", message_metadata={})
    msg = _db_message_to_chat_message(row)
    assert msg.answer_derived_from_carryover is False
    assert msg.findings_block_hops == 0
    assert msg.findings_block is None
    assert msg.turn_summary is None
