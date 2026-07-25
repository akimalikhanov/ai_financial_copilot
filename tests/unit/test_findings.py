"""Unit tests for FindingsLedger (step 10a — the loop-populated scratchpad).

Covers Contract C4 (best-per-aspect by in-place revision, no comparator), C6 (ungrounded
writes dropped, prior entry intact), degraded serving on an unsealed run, and the
projection round-trips back to AgentFindings/AnalyticalFindings.
"""

from __future__ import annotations

from uuid import uuid4

from src.schemas.agent_findings import (
    AgentFindings,
    AnalyticalFindings,
    EntityFinding,
    Observation,
)
from src.schemas.retrieval import ChunkPromptPayload, RetrievedChunk
from src.services.chat.agent.evidence import EvidenceLedger
from src.services.chat.agent.findings import FindingsLedger


def _seed_evidence(n: int) -> tuple[EvidenceLedger, list[str]]:
    """An EvidenceLedger holding n admitted+labelled chunks; returns (ledger, uuid strs)."""
    ledger = EvidenceLedger()
    chunks: list[RetrievedChunk] = []
    payloads: dict = {}
    for i in range(n):
        cid = uuid4()
        chunks.append(
            RetrievedChunk(
                chunk_id=cid,
                document_id=uuid4(),
                score=float(n - i),
                chunk_index=i,
                page_start=1,
                page_end=1,
                heading_trail=[],
                source="vector",
            )
        )
        payloads[cid] = ChunkPromptPayload(
            chunk_id=cid,
            document_id=chunks[-1].document_id,
            document_name="Doc",
            page_numbers=(1,),
            heading_trail=(),
            prompt_text=f"Excerpt {i}.",
        )
    lookup = ledger.start_lookup()
    ledger.admit(lookup, 0, chunks)
    ledger.assign_labels(chunks, payloads)
    return ledger, [str(c.chunk_id) for c in chunks]


class TestGroundingFilter:
    def test_ungrounded_write_dropped_prior_entry_intact(self) -> None:
        # C6: a positive claim whose chunks don't resolve is dropped, leaving the prior
        # grounded entry for that key untouched.
        evidence, ids = _seed_evidence(1)
        ledger = FindingsLedger()

        grounded = EntityFinding(entity="Acme", available=True, value=10.0, source_chunks=[ids[0]])
        assert ledger.record("Acme", grounded, evidence) is True

        ungrounded = EntityFinding(
            entity="Acme", available=True, value=999.0, source_chunks=[str(uuid4())]
        )
        assert ledger.record("Acme", ungrounded, evidence) is False

        served = ledger.projection()
        assert isinstance(served, AgentFindings)
        assert [f.value for f in served.findings] == [10.0]

    def test_negative_finding_admitted_without_chunks(self) -> None:
        # A "not available" finding legitimately cites nothing and must not be dropped.
        evidence, _ = _seed_evidence(1)
        ledger = FindingsLedger()
        stub = EntityFinding(entity="Globex", available=False, source_chunks=[])
        assert ledger.record("Globex", stub, evidence) is True


class TestRevision:
    def test_revision_supersedes_in_place(self) -> None:
        # C4: same key updates in place (best-per-aspect), no duplicate entries, and the
        # revision counter tracks the update.
        evidence, ids = _seed_evidence(1)
        ledger = FindingsLedger()
        first = Observation(
            aspect="margin", claim="Margins fell", evidence_chunks=[ids[0]], confidence="low"
        )
        second = Observation(
            aspect="margin",
            claim="Margins fell sharply",
            evidence_chunks=[ids[0]],
            confidence="high",
        )
        ledger.record("margin", first, evidence)
        ledger.record("margin", second, evidence)

        assert ledger.keys() == {"margin"}
        assert ledger.revised_keys(min_revisions=1) == ["margin"]
        served = ledger.projection()
        assert isinstance(served, AnalyticalFindings)
        assert [o.claim for o in served.observations] == ["Margins fell sharply"]

    def test_first_write_is_not_a_revision(self) -> None:
        evidence, ids = _seed_evidence(1)
        ledger = FindingsLedger()
        ledger.record(
            "margin",
            Observation(aspect="margin", claim="c", evidence_chunks=[ids[0]], confidence="high"),
            evidence,
        )
        assert ledger.revised_keys(min_revisions=1) == []


class TestProjection:
    def test_ingest_agent_round_trips(self) -> None:
        evidence, ids = _seed_evidence(2)
        candidate = AgentFindings(
            metric_requested="revenue",
            comparison_op="argmax",
            findings=(
                EntityFinding(entity="Acme", available=True, value=1.0, source_chunks=[ids[0]]),
                EntityFinding(entity="Globex", available=True, value=2.0, source_chunks=[ids[1]]),
            ),
        )
        ledger = FindingsLedger()
        ledger.ingest(candidate, evidence)
        served = ledger.projection()
        assert isinstance(served, AgentFindings)
        assert served.metric_requested == "revenue"
        assert served.comparison_op == "argmax"
        assert {f.entity for f in served.findings} == {"Acme", "Globex"}

    def test_empty_ledger_projects_none(self) -> None:
        assert FindingsLedger().projection() is None

    def test_reformulated_attempt_does_not_resurrect_dropped_key(self) -> None:
        # A finalizer is a full re-statement: a key present in a rejected attempt but
        # omitted from the reformulation was abandoned, not revised, so it must not leak
        # into the served projection (matches transcript.py stripping rejected drafts).
        evidence, ids = _seed_evidence(2)
        ledger = FindingsLedger()
        ledger.ingest(  # attempt 1 (later rejected): keeps A, plus B it will abandon
            AnalyticalFindings(
                question="q",
                observations=(
                    Observation(
                        aspect="A", claim="keep", evidence_chunks=[ids[0]], confidence="high"
                    ),
                    Observation(
                        aspect="B", claim="abandon", evidence_chunks=[ids[1]], confidence="low"
                    ),
                ),
            ),
            evidence,
        )
        ledger.ingest(  # attempt 2 (accepted): A revised, B dropped, C added
            AnalyticalFindings(
                question="q",
                observations=(
                    Observation(
                        aspect="A",
                        claim="keep revised",
                        evidence_chunks=[ids[0]],
                        confidence="high",
                    ),
                    Observation(
                        aspect="C", claim="new", evidence_chunks=[ids[1]], confidence="high"
                    ),
                ),
            ),
            evidence,
        )
        served = ledger.projection()
        assert isinstance(served, AnalyticalFindings)
        assert {o.aspect for o in served.observations} == {"A", "C"}
        claims = {o.claim for o in served.observations}
        assert "abandon" not in claims  # the dropped key is not resurrected
        assert "keep revised" in claims  # the restated key supersedes in place
        assert ledger.revised_keys(min_revisions=1) == ["A"]


class TestDegradedServing:
    def test_unsealed_ledger_serves_degraded(self) -> None:
        # An unsealed run still serves accumulated findings, with the degraded caveat
        # appended to gaps on the analytical path (subsumes P1-6 best-effort).
        evidence, ids = _seed_evidence(1)
        ledger = FindingsLedger()
        ledger.ingest(
            AnalyticalFindings(
                question="q",
                observations=(
                    Observation(
                        aspect="a", claim="c", evidence_chunks=[ids[0]], confidence="medium"
                    ),
                ),
            ),
            evidence,
        )
        served = ledger.projection(degraded=True)
        assert isinstance(served, AnalyticalFindings)
        assert served.observations  # content is served, not None
        assert any("did not fully converge" in g for g in served.gaps or [])

    def test_sealed_projection_has_no_caveat(self) -> None:
        evidence, ids = _seed_evidence(1)
        ledger = FindingsLedger()
        ledger.ingest(
            AnalyticalFindings(
                question="q",
                observations=(
                    Observation(aspect="a", claim="c", evidence_chunks=[ids[0]], confidence="high"),
                ),
            ),
            evidence,
        )
        served = ledger.projection(degraded=False)
        assert isinstance(served, AnalyticalFindings)
        assert served.gaps is None
