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
from src.services.chat.agent.state import AgentLoopMeta


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


@pytest.fixture(autouse=True)
def _stub_payload_hydration(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_get_chunk_prompt_payloads(_session, chunk_ids):
        return {
            cid: ChunkPromptPayload(
                chunk_id=cid,
                document_id=uuid4(),
                document_name="Doc.pdf",
                page_numbers=(1,),
                heading_trail=("Section",),
                prompt_text="[header]\nSome chunk text.",
            )
            for cid in chunk_ids
        }

    monkeypatch.setattr(synthesis, "get_chunk_prompt_payloads", _fake_get_chunk_prompt_payloads)


class TestNoFindings:
    async def test_no_findings_returns_full_registry_and_no_findings_block(self) -> None:
        c1, c2 = _chunk(), _chunk()
        registry = {c1.chunk_id: c1, c2.chunk_id: c2}

        result = await synthesis.run_synthesis(
            registry,
            None,
            _meta(),
            None,
            None,
            session=None,  # type: ignore[arg-type]
        )

        assert result.findings is None
        assert result.processed is None
        assert result.rag_context.chunk_count == 2
        assert "[STRUCTURED FINDINGS]" not in result.synthesis_context
        assert "[AGENT OBSERVATIONS]" not in result.synthesis_context


class TestCitedChunkNarrowing:
    async def test_narrows_to_cited_chunks(self) -> None:
        cited, uncited = _chunk(), _chunk()
        registry = {cited.chunk_id: cited, uncited.chunk_id: uncited}
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
            registry,
            findings,
            _meta(frozenset({"Acme"})),
            None,
            None,
            session=None,  # type: ignore[arg-type]
        )

        assert {item.chunk_id for item in result.rag_context.items} == {cited.chunk_id}

    async def test_empty_citations_falls_back_to_full_registry(self) -> None:
        c1, c2 = _chunk(), _chunk()
        registry = {c1.chunk_id: c1, c2.chunk_id: c2}
        findings = AgentFindings(
            metric_requested="revenue",
            findings=(EntityFinding(entity="Acme", available=False, reason="not found"),),
        )

        result = await synthesis.run_synthesis(
            registry,
            findings,
            _meta(frozenset({"Acme"})),
            None,
            None,
            session=None,  # type: ignore[arg-type]
        )

        assert {item.chunk_id for item in result.rag_context.items} == {c1.chunk_id, c2.chunk_id}


class TestStubInjection:
    async def test_stub_injected_for_unsearched_entity_with_correct_reason(self) -> None:
        c1 = _chunk()
        registry = {c1.chunk_id: c1}
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
            registry,
            findings,
            _meta(frozenset({"Acme"})),
            scope_result,
            None,
            session=None,  # type: ignore[arg-type]
        )

        assert isinstance(result.findings, AgentFindings)
        stub = next(f for f in result.findings.findings if f.entity == "Globex")
        assert stub.available is False
        assert stub.reason == "not searched by agent"

    async def test_no_stub_for_searched_but_unreported_entity(self) -> None:
        """P1-0: an entity the agent searched but didn't report must not be mislabeled."""
        c1 = _chunk()
        registry = {c1.chunk_id: c1}
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
            registry,
            findings,
            _meta(frozenset({"Acme", "Globex"})),
            scope_result,
            None,
            session=None,  # type: ignore[arg-type]
        )

        assert isinstance(result.findings, AgentFindings)
        assert {f.entity for f in result.findings.findings} == {"Acme"}


class TestSynthesisContextShape:
    async def test_equals_findings_block_plus_formatted_context(self) -> None:
        c1 = _chunk()
        registry = {c1.chunk_id: c1}
        findings = AnalyticalFindings(
            question="Is revenue growing?",
            observations=(
                Observation(
                    claim="Revenue grew",
                    evidence_chunks=[str(c1.chunk_id)],
                    confidence="high",
                ),
            ),
        )

        result = await synthesis.run_synthesis(
            registry,
            findings,
            _meta(),
            None,
            None,
            session=None,  # type: ignore[arg-type]
        )

        expected = (
            "[AGENT OBSERVATIONS]" in result.synthesis_context
            and result.rag_context.formatted_context in result.synthesis_context
        )
        assert expected
        assert result.synthesis_context.endswith(result.rag_context.formatted_context)
