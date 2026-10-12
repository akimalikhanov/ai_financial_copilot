"""The transcript is append-only: no earlier message is rewritten, so the provider can
cache the whole prefix and every label the model can cite stays readable on screen."""

from __future__ import annotations

import json
from uuid import uuid4

import pytest

from src.schemas.retrieval import ChunkPromptPayload, RetrievedChunk
from src.services.chat.agent.evidence import EvidenceLedger
from src.services.chat.agent.transcript import (
    Transcript,
    assistant_msg_with_tool_calls,
    with_status,
)
from src.services.llm_adapters.base_adapter import ChatMessage, Role, ToolCallRef


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        score=1.0,
        chunk_index=0,
        page_start=1,
        page_end=1,
        heading_trail=[],
        source="vector",
    )


def _payload(chunk: RetrievedChunk) -> ChunkPromptPayload:
    return ChunkPromptPayload(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        document_name="Doc.pdf",
        page_numbers=(1,),
        heading_trail=("Section",),
        prompt_text=f"[header]\nchunk {chunk.chunk_id} text.",
    )


def _run_turn(
    ledger: EvidenceLedger, transcript: Transcript, iteration: int, chunks: list[RetrievedChunk]
) -> list[str]:
    """Simulate one search turn: admit chunks, render a tool result, append it. Returns
    the S-labels this turn showed the model."""
    ledger.admit(chunks)
    ctx = ledger.assign_labels(chunks, {c.chunk_id: _payload(c) for c in chunks})
    tc = ToolCallRef(
        id=f"call_{iteration}",
        name="search_documents",
        arguments=json.dumps({"entity": "Acme", "query": "revenue"}),
    )
    transcript.append_tool_calls([tc])
    transcript.append(
        ChatMessage(role=Role.tool, tool_call_id=tc.id, content=ctx.formatted_context)
    )
    return [item.ref_id for item in ctx.items]


@pytest.mark.parametrize("num_turns", [1, 3, 8])
def test_every_label_shown_stays_readable_and_resolvable(num_turns: int) -> None:
    ledger = EvidenceLedger()
    transcript = Transcript([ChatMessage(role=Role.system, content="sys")])

    ever_shown: list[str] = []
    for iteration in range(num_turns):
        ever_shown.extend(_run_turn(ledger, transcript, iteration, [_chunk(), _chunk()]))

    on_screen = "\n".join(m.content or "" for m in transcript.messages)
    for label in ever_shown:
        assert f'id="{label}"' in on_screen
    resolved, unresolved = ledger.resolve_refs(ever_shown)
    assert unresolved == []
    assert len(resolved) == len(ever_shown)


def test_earlier_messages_are_never_rewritten() -> None:
    """A rewritten earlier message breaks the provider's prompt cache for everything after
    it, on every turn."""
    ledger = EvidenceLedger()
    transcript = Transcript([ChatMessage(role=Role.system, content="sys")])

    snapshots: list[list[ChatMessage]] = []
    for iteration in range(4):
        _run_turn(ledger, transcript, iteration, [_chunk(), _chunk()])
        snapshots.append(list(transcript.messages))

    for earlier, later in zip(snapshots, snapshots[1:], strict=False):
        assert later[: len(earlier)] == earlier


def test_a_re_returned_chunk_is_rendered_once_per_run() -> None:
    ledger = EvidenceLedger()
    transcript = Transcript([ChatMessage(role=Role.system, content="sys")])
    repeated = _chunk()

    first = _run_turn(ledger, transcript, 0, [repeated])
    second = _run_turn(ledger, transcript, 1, [repeated, _chunk()])

    assert first == ["S1"]
    assert second == ["S2"]
    on_screen = "\n".join(m.content or "" for m in transcript.messages)
    assert on_screen.count('id="S1"') == 1


class TestWithStatus:
    def _messages(self) -> list[ChatMessage]:
        return [
            ChatMessage(role=Role.system, content="sys"),
            ChatMessage(role=Role.user, content="Q"),
            assistant_msg_with_tool_calls([ToolCallRef(id="c1", name="t", arguments="{}")]),
            ChatMessage(role=Role.tool, tool_call_id="c1", content="excerpts"),
        ]

    def test_status_is_appended_to_the_last_tool_result(self) -> None:
        messages = self._messages()
        sent = with_status(messages, "Open: A1 (q)")
        assert len(sent) == len(messages)
        assert sent[-1].role == Role.tool
        assert sent[-1].tool_call_id == "c1"
        assert sent[-1].content == "excerpts\n\n<run_status>\nOpen: A1 (q)\n</run_status>"
        # Appended, not prepended: the stored content stays a prefix, so the cache holds.
        assert (sent[-1].content or "").startswith(messages[-1].content or "")

    def test_stored_messages_are_not_changed(self) -> None:
        messages = self._messages()
        before = list(messages)
        with_status(messages, "Open: A1 (q)")
        assert messages == before

    def test_call_zero_sends_status_as_a_user_message(self) -> None:
        messages = self._messages()[:2]
        sent = with_status(messages, "Open: Acme (Acme)")
        assert sent[:2] == messages
        assert sent[-1] == ChatMessage(
            role=Role.user, content="<run_status>\nOpen: Acme (Acme)\n</run_status>"
        )

    def test_no_status_leaves_the_request_unchanged(self) -> None:
        messages = self._messages()
        assert with_status(messages, None) == messages
        assert with_status(messages[:2], None) == messages[:2]
