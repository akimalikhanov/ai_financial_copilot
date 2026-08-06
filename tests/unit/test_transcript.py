"""Unit tests for evidence-aware compaction (docs/stages/agentic_state_refactor_v2.md step 9).

Contract C1 — the transcript may drop anything; it may never be the sole holder of
anything. After *any* compaction, every `S\\d+` the transcript has ever shown must still
resolve via `EvidenceLedger.resolve_refs`. That invariant is what licenses aggressive
tool-result eviction: the label survives in the ledger regardless of what the transcript
keeps.
"""

from __future__ import annotations

import json
from uuid import uuid4

import pytest

from src.schemas.retrieval import ChunkPromptPayload, RetrievedChunk
from src.services.chat.agent.evidence import EvidenceLedger
from src.services.chat.agent.transcript import Transcript, cap_history
from src.services.llm_adapters.base_adapter import ChatMessage, Role, ToolCallRef

_EVICTED_PREFIX = "[compacted]"


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
    ledger: EvidenceLedger,
    transcript: Transcript,
    iteration: int,
    *,
    entity: str = "Acme Corp",
    query: str = "revenue",
) -> list[str]:
    """Simulate one search turn: admit chunks, render a tool result, append it. Returns
    the S-labels this turn showed the model."""
    chunks = [_chunk(), _chunk()]
    ledger.admit(chunks)
    ctx = ledger.assign_labels(chunks, {c.chunk_id: _payload(c) for c in chunks})

    tc = ToolCallRef(
        id=f"call_{iteration}",
        name="search_documents",
        arguments=json.dumps({"entity": entity, "query": query}),
    )
    transcript.append_tool_calls([tc])
    transcript.append(
        ChatMessage(role=Role.tool, tool_call_id=tc.id, content=ctx.formatted_context)
    )
    return [item.ref_id for item in ctx.items]


@pytest.mark.parametrize("num_turns", [1, 2, 3, 5, 8])
def test_compaction_preserves_resolvability(num_turns: int) -> None:
    """C1: over a sequence of turns each followed by compaction, every ever-shown label
    still resolves against the ledger — even the ones evicted from the transcript."""
    ledger = EvidenceLedger()
    transcript = Transcript(
        [
            ChatMessage(role=Role.system, content="sys"),
            ChatMessage(role=Role.user, content="q"),
        ]
    )

    ever_shown: list[str] = []
    for iteration in range(num_turns):
        ever_shown.extend(_run_turn(ledger, transcript, iteration))
        transcript.compress()

        # After each compaction, every label ever shown must resolve — including those
        # already evicted from the transcript view.
        resolved, unresolved = ledger.resolve_refs(ever_shown)
        assert unresolved == []
        assert len(resolved) == len(ever_shown)


def test_aggressive_eviction_drops_old_tool_results() -> None:
    """The eviction is real, not a no-op: after several turns only the most recent turn's
    rendered context survives; older tool results are replaced by the stub — which is
    exactly why the ledger (not the transcript) must hold the labels."""
    ledger = EvidenceLedger()
    transcript = Transcript([ChatMessage(role=Role.system, content="sys")])

    shown_per_turn: list[list[str]] = []
    for iteration in range(4):
        shown_per_turn.append(_run_turn(ledger, transcript, iteration))
        transcript.compress()

    tool_msgs = [m for m in transcript.messages if m.role == Role.tool]
    # Default keep_last_n_turns=1 → exactly one non-stub tool result remains.
    live = [m for m in tool_msgs if not (m.content or "").startswith(_EVICTED_PREFIX)]
    assert len(live) == 1

    # The one live tool result is the latest turn's; its labels are literally present.
    for label in shown_per_turn[-1]:
        assert label in (live[0].content or "")
    # An earlier turn's labels are gone from the *bulky* view but still resolve.
    resolved, unresolved = ledger.resolve_refs(shown_per_turn[0])
    assert unresolved == []
    assert len(resolved) == len(shown_per_turn[0])


def test_rendered_then_evicted_then_rereturned_chunk_is_readable() -> None:
    """A chunk rendered in turn 1, evicted by compaction, and re-returned by a later
    search must be readable again — otherwise `resolve_refs` still resolves it while the
    model can read it in neither the transcript nor the fresh tool result, so a claim
    could be grounded on text the model never saw.
    """
    ledger = EvidenceLedger()
    transcript = Transcript([ChatMessage(role=Role.system, content="sys")])

    # Turn 0: two chunks rendered, then compacted out of the model's view.
    chunks = [_chunk(), _chunk()]
    payloads = {c.chunk_id: _payload(c) for c in chunks}
    ledger.admit(chunks)
    ctx0 = ledger.assign_labels(chunks, payloads)
    labels = [i.ref_id for i in ctx0.items]
    tc0 = ToolCallRef(
        id="call_0", name="search_documents", arguments=json.dumps({"entity": "A", "query": "q"})
    )
    transcript.append_tool_calls([tc0])
    transcript.append(
        ChatMessage(role=Role.tool, tool_call_id=tc0.id, content=ctx0.formatted_context)
    )

    # A second turn pushes turn 0 past the keep window.
    _run_turn(ledger, transcript, 1)
    transcript.compress(ledger)

    # Gone as a *rendered excerpt* (the stub keeps a label-range breadcrumb, not the text).
    tag = f'id="{labels[0]}"'
    assert not any(tag in (m.content or "") for m in transcript.messages if m.role == Role.tool)

    # Turn 2 re-returns the evicted chunk: it is revived under its ORIGINAL label.
    ledger.admit([chunks[0]])
    ctx2 = ledger.assign_labels([chunks[0]], payloads)

    assert labels[0] in ctx2.formatted_context, "re-returned evicted chunk must be re-emitted"
    assert [i.ref_id for i in ctx2.items] == [labels[0]], "revival must not mint a new label"


def test_still_rendered_chunk_is_not_re_emitted() -> None:
    """A chunk still visible in the transcript is deduped as before — revival applies
    only to chunks compaction actually evicted."""
    ledger = EvidenceLedger()
    chunks = [_chunk()]
    payloads = {c.chunk_id: _payload(c) for c in chunks}
    ledger.admit(chunks)
    ledger.assign_labels(chunks, payloads)

    ctx2 = ledger.assign_labels(chunks, payloads)
    assert ctx2.items == ()
    assert ctx2.formatted_context == ""


def test_evicted_stub_is_informative() -> None:
    """An evicted result keeps a breadcrumb: which search (entity + query) and which
    labels it produced — not a blank 'truncated' marker."""
    ledger = EvidenceLedger()
    transcript = Transcript([ChatMessage(role=Role.system, content="sys")])

    labels_t0 = _run_turn(
        ledger, transcript, 0, entity="Seiko Epson Corporation", query="pension discount rate"
    )
    transcript.compress()
    _run_turn(ledger, transcript, 1)  # a second turn so turn 0 falls before the cutoff
    transcript.compress()

    stub = next(
        m.content or ""
        for m in transcript.messages
        if (m.content or "").startswith(_EVICTED_PREFIX)
    )
    assert "Seiko Epson Corporation" in stub
    assert "pension discount rate" in stub
    # Label span is preserved so the model knows those citations are still live.
    assert f"{labels_t0[0]}–{labels_t0[-1]}" in stub
    assert f"{len(labels_t0)} excerpts" in stub


def test_non_labelled_tool_results_survive_compaction() -> None:
    """Error / rejection notices carry no S-label, so compaction leaves them intact — the
    model never loses *why* a prior finalizer was rejected."""
    ledger = EvidenceLedger()
    transcript = Transcript([ChatMessage(role=Role.system, content="sys")])

    _run_turn(ledger, transcript, 0)
    # A short, label-free tool result (e.g. a rejection reason) in the same old turn.
    reject_note = "report_analytical_findings rejected — close the gap on discount rate."
    transcript.append(ChatMessage(role=Role.tool, tool_call_id="call_reject", content=reject_note))
    transcript.compress()
    _run_turn(ledger, transcript, 1)
    transcript.compress()

    contents = [m.content for m in transcript.messages if m.role == Role.tool]
    assert reject_note in contents  # untouched
    assert any((c or "").startswith(_EVICTED_PREFIX) for c in contents)  # the bulky one went


def test_report_only_turn_is_not_a_compaction_boundary() -> None:
    """10b §4b: only assistant messages issuing a *search* start a turn.

    Reports are non-terminal now and can arrive in their own turn. Counting one as a
    boundary would shift the cutoff and evict the current search's excerpts a turn early —
    the model would then compose claims from a breadcrumb instead of the text.
    """
    ledger = EvidenceLedger()
    transcript = Transcript([ChatMessage(role=Role.system, content="sys")])

    labels = _run_turn(ledger, transcript, 0)
    # A report-only turn: assistant tool_calls carrying no search.
    transcript.append_tool_calls(
        [ToolCallRef(id="r1", name="report_analytical_findings", arguments="{}")]
    )
    transcript.append(ChatMessage(role=Role.tool, tool_call_id="r1", content="Recorded A1."))
    transcript.compress(ledger)

    live = [
        m
        for m in transcript.messages
        if m.role == Role.tool and not (m.content or "").startswith(_EVICTED_PREFIX)
    ]
    # The search's excerpts survive: the report turn did not push them past the cutoff.
    search_result = next(m for m in live if "retrieved_excerpt" in (m.content or ""))
    for label in labels:
        assert label in (search_result.content or "")


# ---------------------------------------------------------------------------
# 10b §4 row 2 — prior conversation is capped, not carried whole
# ---------------------------------------------------------------------------


def _pair(n: int, answer_len: int = 10) -> list[ChatMessage]:
    return [
        ChatMessage(role=Role.user, content=f"q{n}"),
        ChatMessage(role=Role.assistant, content=f"a{n}" * answer_len),
    ]


def test_cap_history_keeps_only_the_last_n_pairs() -> None:
    messages = [*_pair(1), *_pair(2), *_pair(3)]
    kept = cap_history(messages, max_turns=2, max_assistant_chars=1000)
    assert [m.content for m in kept][0] == "q2"
    assert len(kept) == 4


def test_cap_history_starts_at_a_user_message_never_mid_pair() -> None:
    """Slicing to a message count instead of a turn boundary would carry an assistant
    answer whose question was dropped — worse than useless context."""
    messages = [*_pair(1), *_pair(2), *_pair(3)]
    kept = cap_history(messages, max_turns=1, max_assistant_chars=1000)
    assert [m.role for m in kept] == [Role.user, Role.assistant]
    assert kept[0].content == "q3"


def test_cap_history_truncates_assistant_content_only() -> None:
    messages = [ChatMessage(role=Role.user, content="u" * 500), *_pair(1, answer_len=500)]
    kept = cap_history(messages, max_turns=2, max_assistant_chars=100)
    assert kept[0].content == "u" * 500  # a prior question is the only record of what was asked
    assert kept[-1].content == "a1" * 50 + "\u2026"


def test_cap_history_zero_turns_drops_everything() -> None:
    assert cap_history([*_pair(1)], max_turns=0, max_assistant_chars=100) == []
