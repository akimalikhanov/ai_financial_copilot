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
    NamedItem,
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
        # A finalizer is a full re-statement, so on the *accepted* call a key the model
        # omitted was abandoned, not revised, and must not leak into the served projection
        # (matches transcript.py stripping rejected drafts). The prune lives in `prune_to`,
        # which the loop calls only once a candidate has cleared every gate — see
        # `test_rejected_attempt_does_not_prune` for why `ingest` must not do it (P0-1).
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
        assert ledger.prune_to(
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
            )
        ) == {"B"}
        served = ledger.projection()
        assert isinstance(served, AnalyticalFindings)
        assert {o.aspect for o in served.observations} == {"A", "C"}
        claims = {o.claim for o in served.observations}
        assert "abandon" not in claims  # the dropped key is not resurrected
        assert "keep revised" in claims  # the restated key supersedes in place
        assert ledger.revised_keys(min_revisions=1) == ["A"]

    def test_rejected_attempt_does_not_prune(self) -> None:
        # P0-1: pruning on every fold destroyed established content. When a gate coerces a
        # restatement, the model re-emits from a transcript whose rejected draft has been
        # stripped; a key it fails to reproduce — or reproduces under a renamed aspect —
        # was lost, not abandoned. Keeping it is what lets the gate see the loss at all,
        # and what lets the rejection message hand the content back (P0-2).
        evidence, ids = _seed_evidence(2)
        ledger = FindingsLedger()
        ledger.ingest(
            AnalyticalFindings(
                question="q",
                observations=(
                    Observation(
                        aspect="group_revenue_change",
                        claim="net sales fell ¥40 billion",
                        evidence_chunks=[ids[0]],
                        confidence="high",
                    ),
                ),
            ),
            evidence,
        )
        # The rejected retry renames the aspect and vaguens the figure.
        ledger.ingest(
            AnalyticalFindings(
                question="q",
                observations=(
                    Observation(
                        aspect="total_revenue_change",
                        claim="net sales fell",
                        evidence_chunks=[ids[1]],
                        confidence="high",
                    ),
                ),
            ),
            evidence,
        )
        served = ledger.projection()
        assert isinstance(served, AnalyticalFindings)
        assert {o.aspect for o in served.observations} == {
            "group_revenue_change",
            "total_revenue_change",
        }
        assert "net sales fell ¥40 billion" in {o.claim for o in served.observations}

    def test_add_gap_reaches_the_projection(self) -> None:
        # P1-3: when a drop is allowed to stand, the omission must surface as a stated
        # limitation rather than vanishing.
        evidence, ids = _seed_evidence(1)
        ledger = FindingsLedger()
        ledger.ingest(
            AnalyticalFindings(
                question="q",
                observations=(
                    Observation(
                        aspect="A", claim="keep", evidence_chunks=[ids[0]], confidence="high"
                    ),
                ),
            ),
            evidence,
        )
        ledger.add_gap("Previously reported observations were dropped without resolution: B")
        served = ledger.projection()
        assert isinstance(served, AnalyticalFindings)
        assert served.gaps == [
            "Previously reported observations were dropped without resolution: B"
        ]


class TestItemKeyedObservations:
    """P1-1: the ledger keys observations on aspect *plus* named item, so a discovered set
    of items reported under one shared aspect survives to synthesis intact."""

    @staticmethod
    def _segment(aspect: str, item: str, claim: str, ref: str) -> Observation:
        return Observation(
            aspect=aspect,
            claim=claim,
            evidence_chunks=[ref],
            confidence="high",
            named_item=NamedItem(name=item, status="resolved"),
        )

    def test_three_items_under_one_aspect_are_three_out(self) -> None:
        # The failure this fixes: three in, one out. The gate saw all three (it reads the
        # candidate, pre-collapse) and charged three rejections; synthesis saw one.
        evidence, ids = _seed_evidence(3)
        ledger = FindingsLedger()
        ledger.ingest(
            AnalyticalFindings(
                question="how did each segment perform?",
                observations=(
                    self._segment("segment_performance", "Payments segment", "up 12%", ids[0]),
                    self._segment("segment_performance", "Lending segment", "down 4%", ids[1]),
                    self._segment("segment_performance", "Treasury unit", "flat", ids[2]),
                ),
            ),
            evidence,
        )
        served = ledger.projection()
        assert isinstance(served, AnalyticalFindings)
        assert len(served.observations) == 3
        assert {o.claim for o in served.observations} == {"up 12%", "down 4%", "flat"}

    def test_same_item_still_updates_in_place(self) -> None:
        # C4 must not regress: the same item under the same aspect is one conclusion,
        # revised — including across the unresolved -> resolved transition (D14).
        evidence, ids = _seed_evidence(1)
        ledger = FindingsLedger()
        base = AnalyticalFindings(
            question="q",
            observations=(
                Observation(
                    aspect="segment_mix",
                    claim="Payments was cited; its margin is not stated.",
                    evidence_chunks=[ids[0]],
                    confidence="medium",
                    named_item=NamedItem(name="Payments segment", status="unresolved"),
                ),
            ),
        )
        ledger.ingest(base, evidence)
        ledger.ingest(
            base.model_copy(
                update={
                    "observations": (
                        self._segment(
                            "segment_mix", "the Payments segment", "margin was 22%", ids[0]
                        ),
                    )
                }
            ),
            evidence,
        )
        served = ledger.projection()
        assert isinstance(served, AnalyticalFindings)
        assert len(served.observations) == 1
        assert served.observations[0].claim == "margin was 22%"
        assert ledger.revised_keys(min_revisions=1) == list(ledger.keys())


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
