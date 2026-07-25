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
from src.services.chat.agent.transcript import Transcript
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
    lookup = ledger.start_lookup()
    ledger.admit(lookup, iteration, chunks)
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
