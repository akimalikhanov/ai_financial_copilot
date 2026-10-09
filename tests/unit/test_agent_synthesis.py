"""Unit tests for the synthesis boundary (src.services.chat.agent.synthesis).

Both tasks.py and pipeline_agent.py call run_synthesis, so a test here covers both.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from src.schemas.agent_findings import AgentFindings
from src.schemas.retrieval import ChunkPromptPayload, RetrievedChunk
from src.services.chat.agent import synthesis
from src.services.chat.agent.evidence import EvidenceLedger
from src.services.chat.agent.state import AgentLoopMeta
from src.services.retrieval import payload_hydrator
from tests.unit.test_findings import _f, _fig, _neg


def _chunk(score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        score=score,
        chunk_index=0,
        page_start=1,
        page_end=1,
        heading_trail=[],
        source="vector",
    )


def _meta(convergence_reason: str = "natural") -> AgentLoopMeta:
    return AgentLoopMeta(
        iterations=1,
        tool_calls_total=1,
        convergence_reason=convergence_reason,  # type: ignore[arg-type]
    )


def _payload_for(c: RetrievedChunk) -> ChunkPromptPayload:
    return ChunkPromptPayload(
        chunk_id=c.chunk_id,
        document_id=c.document_id,
        document_name="Doc.pdf",
        page_numbers=(1,),
        heading_trail=("Section",),
        prompt_text="[__REF__]\nSome chunk text.",
    )


def _ledger(*chunks: RetrievedChunk) -> EvidenceLedger:
    """A ledger with every chunk admitted and rendered by one search — i.e. its payloads
    cached, which is what synthesis reads instead of re-hydrating."""
    ledger = EvidenceLedger()
    ledger.admit(chunks)
    ledger.assign_labels(chunks, {c.chunk_id: _payload_for(c) for c in chunks})
    return ledger


async def _run(
    ledger: EvidenceLedger, findings: AgentFindings | None, meta: AgentLoopMeta | None = None
) -> synthesis.AgentRunResult:
    return await synthesis.run_synthesis(
        ledger, findings, meta or _meta(), None, fallback_max_chunks=100
    )


class TestChunkOrdering:
    def test_orders_by_score_across_turn_boundaries(self) -> None:
        """S-labels are assigned in input order, so ordering must be by score alone —
        a later turn's better chunk outranks an earlier turn's worse one."""
        early_weak = _chunk(score=0.2)
        early_weak.turn_index = 0
        late_strong = _chunk(score=0.9)
        late_strong.turn_index = 2
        early_mid = _chunk(score=0.5)
        early_mid.turn_index = 0

        ordered = _ledger(early_weak, late_strong, early_mid).labelled_chunks()

        assert [c.chunk_id for c in ordered] == [
            late_strong.chunk_id,
            early_mid.chunk_id,
            early_weak.chunk_id,
        ]

    def test_highest_scoring_chunk_gets_s1(self) -> None:
        """The confidence badge reads items[0].score, so S1 must be the global best."""
        early_weak = _chunk(score=0.2)
        early_weak.turn_index = 0
        late_strong = _chunk(score=0.9)
        late_strong.turn_index = 3

        ordered = _ledger(early_weak, late_strong).labelled_chunks()
        assert ordered[0].chunk_id == late_strong.chunk_id


class TestSingleHydration:
    async def test_synthesis_reuses_loop_payloads(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Payloads are hydrated once, at search time. A second round-trip here would be
        one avoidable DB call per request on the critical path."""
        ledger = _ledger(_chunk())

        async def _boom(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("synthesis must not re-hydrate chunk payloads")

        monkeypatch.setattr(payload_hydrator, "get_chunk_prompt_payloads", _boom)

        result = await _run(ledger, None)

        assert result.rag_context.chunk_count == 1
        assert "Some chunk text." in result.rag_context.formatted_context


class TestNoFindings:
    async def test_no_findings_returns_full_registry_and_no_findings_block(self) -> None:
        result = await _run(_ledger(_chunk(), _chunk()), None)

        assert result.findings is None
        assert result.processed is None
        assert result.findings_block is None
        assert result.rag_context.chunk_count == 2
        assert "[FINDINGS]" not in result.synthesis_context


class TestCitedChunkNarrowing:
    async def test_narrows_to_cited_chunks(self) -> None:
        cited, uncited = _chunk(), _chunk()
        findings = AgentFindings(
            findings=(_f("Acme", evidence=[str(cited.chunk_id)], figures=[_fig(1.0)]),)
        )

        result = await _run(_ledger(cited, uncited), findings)

        assert {item.chunk_id for item in result.rag_context.items} == {cited.chunk_id}

    async def test_empty_citations_falls_back_to_full_registry(self) -> None:
        c1, c2 = _chunk(), _chunk()
        findings = AgentFindings(findings=(_neg("Acme"),))

        result = await _run(_ledger(c1, c2), findings)

        assert {item.chunk_id for item in result.rag_context.items} == {c1.chunk_id, c2.chunk_id}


class TestUncitedExcerptsNotice:
    async def test_notice_precedes_fallback_excerpts(self) -> None:
        findings = AgentFindings(findings=(_neg("Acme"),))

        result = await _run(_ledger(_chunk()), findings)

        assert result.findings_block is not None
        assert result.synthesis_context.startswith(result.findings_block + "\n\n")
        excerpts = result.synthesis_context[len(result.findings_block) + 2 :]
        assert excerpts.startswith(synthesis.UNCITED_EXCERPTS_NOTICE + "\n\n")

    async def test_notice_leads_when_there_are_no_findings(self) -> None:
        result = await _run(_ledger(_chunk()), None)
        assert result.synthesis_context.startswith(synthesis.UNCITED_EXCERPTS_NOTICE)

    async def test_no_notice_when_findings_cite_excerpts(self) -> None:
        c1 = _chunk()
        findings = AgentFindings(findings=(_f("Acme", evidence=[str(c1.chunk_id)]),))

        result = await _run(_ledger(c1), findings)

        assert synthesis.UNCITED_EXCERPTS_NOTICE not in result.synthesis_context

    async def test_no_notice_without_excerpts(self) -> None:
        result = await _run(EvidenceLedger(), None)
        assert result.synthesis_context == "(No document context.)"


class TestFallbackSelection:
    async def test_cap_takes_each_search_before_any_second_chunk(self) -> None:
        """Two searches, cap 2: one chunk from each, even though the first search's
        second chunk outscores the second search's best."""
        a1, a2 = _chunk(score=0.9), _chunk(score=0.8)
        b1 = _chunk(score=0.05)  # fusion-scale score from a search whose reranker fell open
        ledger = _ledger(a1, a2)
        ledger.assign_labels([b1], {b1.chunk_id: _payload_for(b1)})

        result = await synthesis.run_synthesis(ledger, None, _meta(), None, fallback_max_chunks=2)

        assert [item.chunk_id for item in result.rag_context.items] == [a1.chunk_id, b1.chunk_id]


class TestOneBlock:
    async def test_unresolved_keys_reach_the_block(self) -> None:
        """A seeded entity nobody reported must not vanish — that turns a 2-entity
        comparison into a confident 1-entity argmax."""
        c1 = _chunk()
        findings = AgentFindings(
            findings=(_f("Acme", evidence=[str(c1.chunk_id)], figures=[_fig(1.0)]),),
            comparison_op="argmax",
            unresolved=("Not searched: Globex",),
        )

        result = await _run(_ledger(c1), findings)

        assert result.findings_block is not None
        assert "Unresolved: Not searched: Globex" in result.findings_block

    async def test_figures_are_rendered_under_their_finding(self) -> None:
        c1 = _chunk()
        findings = AgentFindings(
            findings=(
                _f(
                    "Acme",
                    evidence=[str(c1.chunk_id)],
                    figures=[_fig(10.0), _fig(8.0, period_end="2022-12-31")],
                ),
            )
        )

        result = await _run(_ledger(c1), findings)

        assert result.findings_block is not None
        lines = result.findings_block.splitlines()
        i = next(n for n, line in enumerate(lines) if line.startswith("1. Acme "))
        assert lines[i].endswith("| evidence: [S1]")
        # The fixture excerpt states neither number, so both carry the grounding marker.
        assert lines[i + 1].startswith("   - revenue (2023-12-31): USD 10.0M | ⚠ UNVERIFIED")
        assert lines[i + 2].startswith("   - revenue (2022-12-31): USD 8.0M | ⚠ UNVERIFIED")


class TestSynthesisContextShape:
    async def test_equals_findings_block_plus_formatted_context(self) -> None:
        c1 = _chunk()
        findings = AgentFindings(findings=(_f("A1", evidence=[str(c1.chunk_id)]),))

        result = await _run(_ledger(c1), findings)

        assert result.findings_block is not None
        assert result.synthesis_context == (
            result.findings_block + "\n\n" + result.rag_context.formatted_context
        )


class TestRetrievalUnavailable:
    """A dead retrieval backend must not be served as an empty corpus.

    The failure this guards: every search errored, the loop stopped `search_unavailable`
    with zero chunks, and synthesis answered "not found in the uploaded documents" —
    telling the user their documents lack a fact they may contain.
    """

    async def test_banner_leads_context_when_every_search_errored(self) -> None:
        result = await _run(EvidenceLedger(), None, _meta("search_unavailable"))
        assert result.synthesis_context.startswith(synthesis.RETRIEVAL_UNAVAILABLE_BANNER)

    async def test_banner_absent_on_a_genuinely_empty_corpus(self) -> None:
        """Same zero-chunk shape, but the searches ran — this one really is "not found"."""
        result = await _run(EvidenceLedger(), None, _meta("convergence"))
        assert synthesis.RETRIEVAL_UNAVAILABLE_BANNER not in result.synthesis_context

    async def test_banner_precedes_evidence_gathered_before_the_failure(self) -> None:
        """Partial findings are reframed, not discarded: a run that landed chunks and
        then lost the backend still serves what it has, under the banner."""
        result = await _run(_ledger(_chunk()), None, _meta("search_unavailable"))

        assert result.synthesis_context.startswith(synthesis.RETRIEVAL_UNAVAILABLE_BANNER)
        assert "Some chunk text." in result.synthesis_context
