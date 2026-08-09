"""Contract F1: carried findings must never reach the agent's transcript.

The agent sees user turns and synthesis prose. It must not see a prior turn's findings
block, nor prose derived from one — such a number has no plan entry, no AspectStats, and
no ledger backing it.
"""

from __future__ import annotations

import dataclasses

import pytest

from src.models.message import Message, MessageRole
from src.schemas import chat as schemas
from src.services.chat.agent.loop import CARRYOVER_STUB, build_agent_history
from src.services.chat.agent.transcript import cap_history
from src.services.context.conversation_history import _db_message_to_chat_message
from src.services.llm_adapters.base_adapter import ChatMessage as AdapterChatMessage

SENTINEL = "SENTINEL_FINDINGS_BLOCK_LEAK_CANARY"

BLOCK = f"""[STRUCTURED FINDINGS]
Metric: net income | Operation: argmin
Acme Corp             | USD -97,157.0K | native | period: 2022-12-31 | chunks: {SENTINEL}
[END STRUCTURED FINDINGS]"""


def _history() -> list[schemas.ChatMessage]:
    return [
        schemas.ChatMessage(role=schemas.Role.user, content="what was net income?"),
        schemas.ChatMessage(
            role=schemas.Role.assistant,
            content="Acme reported a net loss of USD 97,157K.",
            findings_block=BLOCK,
        ),
    ]


def test_adapter_message_cannot_hold_a_findings_block() -> None:
    """The boundary is structural: the adapter type has no such field and is slotted."""
    fields = {f.name for f in dataclasses.fields(AdapterChatMessage)}
    assert "findings_block" not in fields
    assert getattr(AdapterChatMessage, "__dataclass_params__").frozen  # noqa: B009
    assert hasattr(AdapterChatMessage, "__slots__")

    # pyright flags this call itself — that is the contract holding at type-check time.
    with pytest.raises(TypeError):
        AdapterChatMessage(role=schemas.Role.user, content="x", findings_block=SENTINEL)  # type: ignore[call-arg]


def test_findings_block_does_not_reach_agent_history() -> None:
    built = build_agent_history(_history(), scan=False)
    assert SENTINEL not in repr(built)
    # The assistant's own prose still crosses — that is intended.
    assert any("net loss of USD 97,157K" in (m.content or "") for m in built)


def test_findings_block_survives_cap_history() -> None:
    """cap_history rebuilds messages; assert nothing smuggles the block back in."""
    built = cap_history(
        build_agent_history(_history(), scan=False),
        max_turns=5,
        max_assistant_chars=10_000,
    )
    assert SENTINEL not in repr(built)


def test_carryover_derived_prose_is_stubbed() -> None:
    """Step 7: a direct_answer turn restates earlier numbers, so the agent gets a stub."""
    history = [
        schemas.ChatMessage(role=schemas.Role.user, content="summarize as a table"),
        schemas.ChatMessage(
            role=schemas.Role.assistant,
            content="| Metric | Value |\n| Net income | USD -97,157K |",
            answer_derived_from_carryover=True,
        ),
    ]
    built = build_agent_history(history, scan=False)

    assert "97,157" not in repr(built)
    assert any(m.content == CARRYOVER_STUB for m in built)
    # The turn is still present, so user/assistant pairing stays intact for cap_history.
    assert [m.role.value for m in built] == ["user", "assistant"]


def test_normal_assistant_prose_is_not_stubbed() -> None:
    built = build_agent_history(_history(), scan=False)
    assert all(m.content != CARRYOVER_STUB for m in built)


def test_stub_holds_with_injection_scanning_on() -> None:
    """Scanning rewrites user turns; the assistant stub must be unaffected by it."""
    history = [
        schemas.ChatMessage(role=schemas.Role.user, content="summarize as a table"),
        schemas.ChatMessage(
            role=schemas.Role.assistant,
            content="| Net income | USD -97,157K |",
            answer_derived_from_carryover=True,
        ),
    ]
    built = build_agent_history(history, scan=True)
    assert "97,157" not in repr(built)
    assert any(m.content == CARRYOVER_STUB for m in built)


def test_carryover_flag_survives_the_cache_round_trip() -> None:
    """The stub only fires if the flag comes back off the Redis tail."""
    original = schemas.ChatMessage(
        role=schemas.Role.assistant,
        content="| Net income | USD -97,157K |",
        answer_derived_from_carryover=True,
        findings_block=BLOCK,
        findings_block_hops=2,
    )
    restored = schemas.ChatMessage.model_validate(original.model_dump(mode="json"))
    assert restored.answer_derived_from_carryover is True
    assert restored.findings_block_hops == 2

    built = build_agent_history([restored], scan=False)
    assert SENTINEL not in repr(built)
    assert built[0].content == CARRYOVER_STUB


def test_legacy_cache_entry_defaults_to_not_carried() -> None:
    """Pre-v2 entries lack the new keys; they must read as ordinary prose."""
    legacy = schemas.ChatMessage.model_validate({"role": "assistant", "content": "old answer"})
    assert legacy.answer_derived_from_carryover is False
    assert legacy.findings_block_hops == 0
    assert legacy.findings_block is None
    assert build_agent_history([legacy], scan=False)[0].content == "old answer"


def test_cold_cache_db_path_matches_the_cache_path() -> None:
    """On TTL expiry history is rebuilt from Postgres — it must carry the same fields,
    or the stub silently stops firing once the cache drops."""
    row = Message(
        role=MessageRole.assistant,
        content="| Net income | USD -97,157K |",
        message_metadata={
            "findings_block": BLOCK,
            "answer_derived_from_carryover": True,
            "findings_block_hops": 2,
        },
    )
    msg = _db_message_to_chat_message(row)
    assert msg.answer_derived_from_carryover is True
    assert msg.findings_block_hops == 2
    assert msg.findings_block == BLOCK

    built = build_agent_history([msg], scan=False)
    assert SENTINEL not in repr(built)
    assert built[0].content == CARRYOVER_STUB


def test_cold_cache_legacy_row_defaults() -> None:
    row = Message(role=MessageRole.assistant, content="old answer", message_metadata={})
    msg = _db_message_to_chat_message(row)
    assert msg.answer_derived_from_carryover is False
    assert msg.findings_block_hops == 0
    assert msg.findings_block is None
