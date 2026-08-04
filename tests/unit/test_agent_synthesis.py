"""Unit tests for the synthesis boundary (src.services.chat.agent.synthesis).

Covers the anti-drift guard from docs/stages/agentic_state_refactor_v2.md step 3: both
tasks.py and pipeline_agent.py now call run_synthesis, so a test here covers both.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from src.schemas.agent_findings import AgentFindings, AnalyticalFindings, EntityFinding, Observation
from src.schemas.query_router import DocumentScopeResult
from src.schemas.retrieval import ChunkPromptPayload, RetrievedChunk
from src.services.chat.agent import synthesis
from src.services.chat.agent.evidence import EvidenceLedger
from src.services.chat.agent.state import AgentLoopMeta
from src.services.retrieval import payload_hydrator


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


def _meta(searched_entities: frozenset[str] = frozenset()) -> AgentLoopMeta:
    return AgentLoopMeta(
        iterations=1,
        tool_calls_total=1,
        convergence_reason="natural",
        searched_entities=searched_entities,
    )


def _ledger(*chunks: RetrievedChunk) -> EvidenceLedger:
    """A ledger with every chunk admitted and rendered — i.e. its payloads cached, which
    is what synthesis reads instead of re-hydrating (D2)."""
    ledger = EvidenceLedger()
    ledger.admit(chunks)
    ledger.assign_labels(
        chunks,
        {
            c.chunk_id: ChunkPromptPayload(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                document_name="Doc.pdf",
                page_numbers=(1,),
                heading_trail=("Section",),
                prompt_text="[__REF__]\nSome chunk text.",
            )
            for c in chunks
        },
    )
    return ledger


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

        ordered = _ledger(early_weak, late_strong, early_mid).ordered_chunks()

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

        ordered = _ledger(early_weak, late_strong).ordered_chunks()
        assert ordered[0].chunk_id == late_strong.chunk_id


class TestSingleHydration:
    async def test_synthesis_reuses_loop_payloads(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """D2: payloads are hydrated once, at search time. A second round-trip here would
        be one avoidable DB call per request on the critical path."""
        c1 = _chunk()
        ledger = _ledger(c1)

        async def _boom(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("synthesis must not re-hydrate chunk payloads")

        monkeypatch.setattr(payload_hydrator, "get_chunk_prompt_payloads", _boom)

        result = await synthesis.run_synthesis(
            ledger, None, _meta(), None, None, max_chunks_per_entity=100
        )

        assert result.rag_context.chunk_count == 1
        assert "Some chunk text." in result.rag_context.formatted_context


class TestNoFindings:
    async def test_no_findings_returns_full_registry_and_no_findings_block(self) -> None:
        c1, c2 = _chunk(), _chunk()
        ledger = _ledger(c1, c2)

        result = await synthesis.run_synthesis(
            ledger,
            None,
            _meta(),
            None,
            None,
            max_chunks_per_entity=100,
        )

        assert result.findings is None
        assert result.processed is None
        assert result.rag_context.chunk_count == 2
        assert "[STRUCTURED FINDINGS]" not in result.synthesis_context
        assert "[AGENT OBSERVATIONS]" not in result.synthesis_context


class TestCitedChunkNarrowing:
    async def test_narrows_to_cited_chunks(self) -> None:
        cited, uncited = _chunk(), _chunk()
        ledger = _ledger(cited, uncited)
        findings = AgentFindings(
            metric_requested="revenue",
            findings=(
                EntityFinding(
                    entity="Acme",
                    available=True,
                    value=1.0,
                    currency="USD",
                    source_chunks=[str(cited.chunk_id)],
                ),
            ),
        )

        result = await synthesis.run_synthesis(
            ledger,
            findings,
            _meta(frozenset({"Acme"})),
            None,
            None,
            max_chunks_per_entity=100,
        )

        assert {item.chunk_id for item in result.rag_context.items} == {cited.chunk_id}

    async def test_empty_citations_falls_back_to_full_registry(self) -> None:
        c1, c2 = _chunk(), _chunk()
        ledger = _ledger(c1, c2)
        findings = AgentFindings(
            metric_requested="revenue",
            findings=(EntityFinding(entity="Acme", available=False, reason="not found"),),
        )

        result = await synthesis.run_synthesis(
            ledger,
            findings,
            _meta(frozenset({"Acme"})),
            None,
            None,
            max_chunks_per_entity=100,
        )

        assert {item.chunk_id for item in result.rag_context.items} == {c1.chunk_id, c2.chunk_id}


class TestStubInjection:
    async def test_stub_injected_for_unsearched_entity_with_correct_reason(self) -> None:
        c1 = _chunk()
        ledger = _ledger(c1)
        findings = AgentFindings(
            metric_requested="revenue",
            findings=(
                EntityFinding(
                    entity="Acme", available=True, value=1.0, source_chunks=[str(c1.chunk_id)]
                ),
            ),
        )
        scope_result = DocumentScopeResult(
            source="entity_resolved",
            doc_ids=None,
            per_entity_doc_ids={"Acme": [uuid4()], "Globex": [uuid4()]},
        )

        result = await synthesis.run_synthesis(
            ledger,
            findings,
            _meta(frozenset({"Acme"})),
            scope_result,
            None,
            max_chunks_per_entity=100,
        )

        assert isinstance(result.findings, AgentFindings)
        stub = next(f for f in result.findings.findings if f.entity == "Globex")
        assert stub.available is False
        assert stub.reason == "not searched by agent"

    async def test_no_stub_for_searched_but_unreported_entity(self) -> None:
        """P1-0: an entity the agent searched but didn't report must not be mislabeled."""
        c1 = _chunk()
        ledger = _ledger(c1)
        findings = AgentFindings(
            metric_requested="revenue",
            findings=(
                EntityFinding(
                    entity="Acme", available=True, value=1.0, source_chunks=[str(c1.chunk_id)]
                ),
            ),
        )
        scope_result = DocumentScopeResult(
            source="entity_resolved",
            doc_ids=None,
            per_entity_doc_ids={"Acme": [uuid4()], "Globex": [uuid4()]},
        )

        result = await synthesis.run_synthesis(
            ledger,
            findings,
            _meta(frozenset({"Acme", "Globex"})),
            scope_result,
            None,
            max_chunks_per_entity=100,
        )

        assert isinstance(result.findings, AgentFindings)
        assert {f.entity for f in result.findings.findings} == {"Acme"}


class TestSynthesisContextShape:
    async def test_equals_findings_block_plus_formatted_context(self) -> None:
        c1 = _chunk()
        ledger = _ledger(c1)
        findings = AnalyticalFindings(
            question="Is revenue growing?",
            observations=(
                Observation(
                    aspect="revenue_trend",
                    claim="Revenue grew",
                    evidence_chunks=[str(c1.chunk_id)],
                    confidence="high",
                ),
            ),
        )

        result = await synthesis.run_synthesis(
            ledger,
            findings,
            _meta(),
            None,
            None,
            max_chunks_per_entity=100,
        )

        expected = (
            "[AGENT OBSERVATIONS]" in result.synthesis_context
            and result.rag_context.formatted_context in result.synthesis_context
        )
        assert expected
        assert result.synthesis_context.endswith(result.rag_context.formatted_context)
