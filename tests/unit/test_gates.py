"""Unit tests for finalizer gates (docs/stages/agentic_state_refactor_v2.md step 6)."""

from __future__ import annotations

from uuid import uuid4

import pytest
from fakeredis import FakeAsyncRedis

from src.schemas.agent_findings import (
    AgentFindings,
    AnalyticalFindings,
    EntityFinding,
    NamedItem,
    Observation,
)
from src.schemas.retrieval import ChunkPromptPayload, RetrievedChunk
from src.services.chat.agent import gates as gates_module
from src.services.chat.agent.evidence import EvidenceLedger
from src.services.chat.agent.findings import FindingsLedger
from src.services.chat.agent.gates import _normalize_item_name, unresolved_named_items
from src.services.chat.agent.state import AgentRunState, EffortPrior
from src.services.chat.agent.tools import gates_for
from src.services.chat.agent.transcript import Transcript
from src.services.llm_adapters.base_adapter import LLMResponseStats, ToolCallRef


def _reported(*entities: str) -> FindingsLedger:
    """A FindingsLedger keyed on the given entities (available=false is grounding-exempt,
    so no EvidenceLedger seeding is needed to give each a key)."""
    ledger = FindingsLedger()
    for name in entities:
        ledger.record(name, EntityFinding(entity=name, available=False), EvidenceLedger())
    return ledger


def _state(**overrides: object) -> AgentRunState:
    defaults: dict[str, object] = {
        "effort": EffortPrior(
            max_iterations=4,
            max_empty_rounds=1,
            max_insufficiency_rejections=1,
            max_concurrent_searches=3,
            max_named_item_rejections_per_item=2,
            max_named_item_rejections_total=10,
        ),
        "token_budget": 1_000_000,
        "turn_timeout_seconds": 60.0,
        "transcript": Transcript([]),
        "evidence": EvidenceLedger(),
        "expected_entities": set(),
    }
    defaults.update(overrides)
    return AgentRunState(**defaults)  # type: ignore[arg-type]


def _obs(
    aspect: str,
    *,
    claim: str = "a claim",
    item: str | None = None,
    status: str = "unresolved",
    confidence: str = "high",
    ref: str = "S1",
) -> Observation:
    """An observation, optionally carrying a named item.

    `ref` defaults to the S-label form the model emits; ledger-level tests pass the
    resolved chunk-UUID string instead, since that is what `ingest` sees in practice.
    """
    return Observation(
        aspect=aspect,
        claim=claim,
        evidence_chunks=[ref],
        confidence=confidence,  # type: ignore[arg-type]
        named_item=NamedItem(name=item, status=status) if item else None,  # type: ignore[arg-type]
    )


class TestNormalizeItemName:
    def test_case_punctuation_and_leading_article_collapse(self) -> None:
        # AC-5/AC-10: the deterministic normalization D9 specifies, and no more.
        assert _normalize_item_name("the Payments segment") == "payments segment"
        assert _normalize_item_name("Payments segment") == "payments segment"
        assert _normalize_item_name("  'Payments  segment.'  ") == "payments segment"
        assert _normalize_item_name("A Payments segment") == "payments segment"
        assert _normalize_item_name("An Acme unit") == "acme unit"

    def test_only_one_leading_article_is_dropped(self) -> None:
        assert _normalize_item_name("the a team") == "a team"

    def test_article_prefix_of_a_word_is_not_stripped(self) -> None:
        # "Theta" starts with "the" but is not a leading article.
        assert _normalize_item_name("Theta segment") == "theta segment"

    def test_no_similarity_matching(self) -> None:
        # FR-12/EC-5: wording differing by more than case/punctuation/article stays distinct.
        assert _normalize_item_name("Payments") != _normalize_item_name("the payments division")


class TestUnresolvedNamedItems:
    def test_no_named_items_yields_empty(self) -> None:
        # AC-1, EC-8: nothing to chase when no observation names an item.
        assert unresolved_named_items([_obs("cogs"), _obs("margin")]) == {}

    def test_unresolved_item_is_collected_with_model_wording(self) -> None:
        # D15: display name and aspect come from the unresolved-reporting observation.
        items = unresolved_named_items([_obs("payments_rev", item="the Payments segment")])
        assert items == {"payments segment": ("the Payments segment", "payments_rev")}

    def test_variant_spellings_collapse_to_one_key(self) -> None:
        # AC-5, AC-10.
        items = unresolved_named_items(
            [
                _obs("a1", item="the Payments segment"),
                _obs("a2", item="Payments segment"),
            ]
        )
        assert list(items) == ["payments segment"]

    def test_resolved_suppresses_unresolved_in_either_order(self) -> None:
        # AC-3, EC-2, FR-8: resolution wins regardless of observation order.
        unresolved_first = [
            _obs("a1", item="Payments segment"),
            _obs("a1", item="Payments segment", status="resolved"),
        ]
        assert unresolved_named_items(unresolved_first) == {}
        assert unresolved_named_items(list(reversed(unresolved_first))) == {}

    def test_confirmed_absent_suppresses_in_either_order(self) -> None:
        # AC-11: a known absence closes the item just as a resolution does.
        obs = [
            _obs("a1", item="Payments segment"),
            _obs("a1", item="Payments segment", status="confirmed_absent"),
        ]
        assert unresolved_named_items(obs) == {}
        assert unresolved_named_items(list(reversed(obs))) == {}

    def test_confirmed_absent_stays_distinguishable_from_resolved(self) -> None:
        # AC-11: both close the item here, but the status itself is preserved on the
        # observation — T-7's renderer reports them differently.
        absent = _obs("a1", item="Payments segment", status="confirmed_absent")
        resolved = _obs("a2", item="Other segment", status="resolved")
        assert absent.named_item is not None and resolved.named_item is not None
        assert absent.named_item.status == "confirmed_absent"
        assert resolved.named_item.status == "resolved"
        assert unresolved_named_items([absent, resolved]) == {}

    def test_two_disagreeing_resolved_values_yield_one_resolved_key(self) -> None:
        # AC-17, EC-11, D12: no value comparison, no conflict logic here.
        obs = [
            _obs(
                "a1", claim="Payments revenue was $12M", item="Payments segment", status="resolved"
            ),
            _obs(
                "a2", claim="Payments revenue was $19M", item="Payments segment", status="resolved"
            ),
        ]
        assert unresolved_named_items(obs) == {}

    def test_distinct_items_tracked_separately(self) -> None:
        items = unresolved_named_items(
            [
                _obs("a1", item="Payments segment"),
                _obs("a2", item="Lending segment", status="resolved"),
                _obs("a3", item="Treasury unit"),
            ]
        )
        assert set(items) == {"payments segment", "treasury unit"}

    def test_blank_name_is_ignored(self) -> None:
        assert unresolved_named_items([_obs("a1", item="   ...   ")]) == {}


def _analytical(*observations: Observation, gaps: list[str] | None = None) -> AnalyticalFindings:
    return AnalyticalFindings(
        question="why did margins move?", observations=observations, gaps=gaps
    )


class TestNamedItemGate:
    def test_unresolved_item_rejects_quoting_name_and_aspect(self) -> None:
        # AC-2: the reason quotes the model's own wording back to it (D15).
        state = _state()
        candidate = _analytical(_obs("revenue_driver", item="the Payments segment"))
        reason = gates_module.named_item_gate(candidate, state)
        assert reason is not None
        assert '"the Payments segment"' in reason
        assert "aspect: revenue_driver" in reason
        assert "search_documents" in reason

    def test_self_contained_observations_allow_and_move_no_counter(self) -> None:
        # AC-1: nothing to chase -> no rejection and no budget consumed.
        state = _state()
        candidate = _analytical(_obs("cogs"), _obs("margin"))
        assert gates_module.named_item_gate(candidate, state) is None
        assert state.named_item_rejections == {}
        assert state.named_item_rejections_total == 0

    def test_resolved_item_allows(self) -> None:
        state = _state()
        candidate = _analytical(_obs("a1", item="Payments segment", status="resolved"))
        assert gates_module.named_item_gate(candidate, state) is None
        assert state.named_item_rejections_total == 0

    def test_agent_findings_candidate_is_ignored(self) -> None:
        # AC-9, FR-11, EC-7: this gate is scoped to the analytical finalizer only.
        state = _state()
        candidate = AgentFindings(metric_requested="revenue", findings=())
        assert gates_module.named_item_gate(candidate, state) is None
        assert state.named_item_rejections_total == 0

    def test_per_item_cap_stops_chasing_the_same_item(self) -> None:
        # AC-12, FR-4a: the count is tied to the rejection event, not to whether a
        # search happened in between — two bare re-reports still charge twice.
        state = _state()
        candidate = _analytical(_obs("a1", item="Payments segment"))
        assert gates_module.named_item_gate(candidate, state) is not None
        assert gates_module.named_item_gate(candidate, state) is not None
        assert state.named_item_rejections["payments segment"] == 2
        # Third attempt: the item is at its cap (2) and no longer causes rejection.
        assert gates_module.named_item_gate(candidate, state) is None
        assert state.named_item_rejections["payments segment"] == 2

    def test_capped_item_does_not_block_a_second_uncapped_item(self) -> None:
        # AC-4: one item exhausting its budget must not silence the gate for another.
        state = _state()
        state.named_item_rejections["payments segment"] = 2
        state.named_item_rejections_total = 2
        candidate = _analytical(
            _obs("a1", item="Payments segment"),
            _obs("a2", item="Treasury unit"),
        )
        reason = gates_module.named_item_gate(candidate, state)
        assert reason is not None
        assert "Treasury unit" in reason
        assert "Payments segment" not in reason

    def test_capped_item_adds_nothing_to_the_total(self) -> None:
        # Pins plan §3's FR-5 reading: only items attributable to this rejection --
        # those under their cap and named in the message -- consume the shared pool.
        state = _state()
        state.named_item_rejections["payments segment"] = 2
        state.named_item_rejections_total = 2
        candidate = _analytical(
            _obs("a1", item="Payments segment"),
            _obs("a2", item="Treasury unit"),
        )
        assert gates_module.named_item_gate(candidate, state) is not None
        assert state.named_item_rejections_total == 3  # +1 for Treasury only, not +2

    def test_newly_appearing_item_starts_its_own_count(self) -> None:
        # AC-6, FR-9: per-item counts are independent; the shared total continues.
        state = _state()
        first = _analytical(_obs("a1", item="Payments segment"))
        assert gates_module.named_item_gate(first, state) is not None
        assert state.named_item_rejections_total == 1

        second = _analytical(_obs("a2", item="Treasury unit"))
        assert gates_module.named_item_gate(second, state) is not None
        assert state.named_item_rejections["treasury unit"] == 1
        assert state.named_item_rejections_total == 2

    def test_one_rejection_with_three_items_adds_three_to_the_total(self) -> None:
        # AC-6, D7: the overall counter counts items, not rejection events.
        state = _state()
        candidate = _analytical(
            _obs("a1", item="Payments segment"),
            _obs("a2", item="Treasury unit"),
            _obs("a3", item="Aurora Holdings"),
        )
        assert gates_module.named_item_gate(candidate, state) is not None
        assert state.named_item_rejections_total == 3
        assert set(state.named_item_rejections) == {
            "payments segment",
            "treasury unit",
            "aurora holdings",
        }

    def test_overall_cap_waives_the_gate(self) -> None:
        state = _state()
        state.named_item_rejections_total = 10  # == max_named_item_rejections_total
        candidate = _analytical(_obs("a1", item="Payments segment"))
        assert gates_module.named_item_gate(candidate, state) is None
        assert state.named_item_rejections == {}

    def test_zero_total_cap_is_the_kill_switch(self) -> None:
        state = _state(
            effort=EffortPrior(
                max_iterations=4,
                max_empty_rounds=1,
                max_insufficiency_rejections=1,
                max_concurrent_searches=3,
                max_named_item_rejections_per_item=2,
                max_named_item_rejections_total=0,
            )
        )
        candidate = _analytical(_obs("a1", item="Payments segment"))
        assert gates_module.named_item_gate(candidate, state) is None

    def test_last_iteration_waives_the_gate(self) -> None:
        # AC-8, EC-4: no point rejecting when no further round can happen.
        state = _state(iteration=3)  # max_iterations=4 -> iteration == max - 1
        candidate = _analytical(_obs("a1", item="Payments segment"))
        assert gates_module.named_item_gate(candidate, state) is None
        assert state.named_item_rejections_total == 0

    def test_exhausted_token_budget_waives_the_gate(self) -> None:
        # AC-8, EC-4: same predicate as analytical_insufficiency_gate.
        state = _state(token_budget=100)
        state.record_spend("model-a", LLMResponseStats(input_tokens=101))
        assert state.spend_within_budget() is False
        candidate = _analytical(_obs("a1", item="Payments segment"))
        assert gates_module.named_item_gate(candidate, state) is None
        assert state.named_item_rejections_total == 0

    def test_ignores_insufficiency_rejections_budget(self) -> None:
        # D1: the two gates hold independent budgets. An insufficiency budget already
        # spent past its cap must not waive this gate.
        state = _state()
        state.insufficiency_rejections = 99
        candidate = _analytical(_obs("a1", item="Payments segment"))
        assert gates_module.named_item_gate(candidate, state) is not None

    def test_the_two_counters_move_independently(self) -> None:
        # D1 across a mixed sequence: this gate never touches insufficiency_rejections,
        # and the insufficiency gate never touches the named-item counters.
        state = _state()
        thin = _analytical(_obs("a1", item="Payments segment"), gaps=["missing segment detail"])

        assert gates_module.named_item_gate(thin, state) is not None
        assert state.named_item_rejections_total == 1
        # Neither the gate nor its rejection touches the other budget — see
        # TestRejectBudgetAttribution for the `reject()` half.
        assert state.insufficiency_rejections == 0

        assert gates_module.analytical_insufficiency_gate(thin, state) is not None
        assert state.named_item_rejections_total == 1  # unmoved by the other gate
        assert state.insufficiency_rejections == 0

    # The projection half of the gate's union is exercised in T-6, which seeds a real
    # admitted chunk into EvidenceLedger — C6's grounding filter drops observations whose
    # refs don't resolve, so an empty ledger yields an empty projection.

    def test_candidate_tripping_both_gates_reports_the_named_item(self) -> None:
        # AC-14, FR-13: registration order decides which reason the loop surfaces, and
        # the insufficiency budget must be left untouched when this gate wins.
        state = _state()
        # gaps present -> analytical_insufficiency_gate would also reject this.
        candidate = _analytical(
            _obs("revenue_driver", item="the Payments segment"), gaps=["segment detail missing"]
        )
        assert gates_module.analytical_insufficiency_gate(candidate, _state()) is not None

        # Mirror loop.py's dispatch: run gates in registration order, stop at the first.
        first_reason = next(
            (
                r
                for r in (g(candidate, state) for g in gates_for("report_analytical_findings"))
                if r is not None
            ),
            None,
        )
        assert first_reason is not None
        assert "the Payments segment" in first_reason
        assert state.insufficiency_rejections == 0


class TestRejectBudgetAttribution:
    """`reject()` charges the budget it is told to, not one inferred from the candidate.

    Inferring it from `isinstance(candidate, AnalyticalFindings)` meant a named-item
    rejection also spent the insufficiency budget. With
    `AGENT_MAX_INSUFFICIENCY_REJECTIONS` defaulting to 1, one find-then-follow round
    silently disabled `analytical_insufficiency_gate` for the rest of the run — a
    regression to pre-existing behaviour, and a direct D1/FR-5 violation.
    """

    @staticmethod
    def _tc() -> ToolCallRef:
        return ToolCallRef(
            id="call_1", name="report_analytical_findings", arguments='{"question":"q"}'
        )

    @pytest.mark.asyncio
    async def test_named_item_rejection_does_not_spend_the_insufficiency_budget(self) -> None:
        state = _state()
        candidate = _analytical(_obs("revenue_driver", item="Payments segment"))
        await gates_module.reject(
            reason="these named items were reported without their value: ...",
            finalizer_tc=self._tc(),
            candidate=candidate,
            state=state,
            redis_app=FakeAsyncRedis(),
            request_id="req-1",
            metric_status="rejected_named_item",
            charge_insufficiency=False,
        )
        assert state.insufficiency_rejections == 0
        # ...and the insufficiency gate is therefore still armed for the next attempt.
        thin = _analytical(_obs("revenue_driver"), gaps=["still missing"])
        assert gates_module.analytical_insufficiency_gate(thin, state) is not None

    @pytest.mark.asyncio
    async def test_insufficiency_rejection_still_spends_its_own_budget(self) -> None:
        """The default path is unchanged — `charge_insufficiency` defaults to True."""
        state = _state()
        candidate = _analytical(_obs("revenue_driver"), gaps=["still missing"])
        await gates_module.reject(
            reason="Open gaps remain: still missing.",
            finalizer_tc=self._tc(),
            candidate=candidate,
            state=state,
            redis_app=FakeAsyncRedis(),
            request_id="req-2",
        )
        assert state.insufficiency_rejections == 1
        assert state.named_item_rejections_total == 0


def _seeded_evidence() -> tuple[EvidenceLedger, str]:
    """An EvidenceLedger holding one admitted, labelled chunk, plus its ref string.

    C6's grounding filter drops any observation whose evidence_chunks don't resolve, so
    an empty ledger makes `ingest` a no-op and every ledger-level assertion vacuous.
    The returned ref is the chunk's **UUID string**, not "S1": by `ingest` time the loop
    has already resolved S-labels to UUIDs, which is what `_resolves` matches on.
    """
    ledger = EvidenceLedger()
    chunk = RetrievedChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        score=1.0,
        chunk_index=0,
        page_start=1,
        page_end=1,
        heading_trail=[],
        source="vector",
    )
    ledger.admit(ledger.start_lookup(), 0, [chunk])
    ledger.assign_labels(
        [chunk],
        {
            chunk.chunk_id: ChunkPromptPayload(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                document_name="Doc.pdf",
                page_numbers=(1,),
                heading_trail=("Section",),
                prompt_text="[header]\nsegment table text.",
            )
        },
    )
    resolved, _ = ledger.resolve_refs(["S1"])
    assert resolved, "fixture must make S1 resolvable"
    return ledger, resolved[0]


class TestNamedItemGateThroughLedger:
    """The gate against real `FindingsLedger.ingest()` + `EvidenceLedger`, rather than
    hand-built observation sequences — the one place the plan's verified collapse
    behaviour is pinned (plan §3's R3)."""

    def test_restated_resolution_survives_then_omission_reflags(self) -> None:
        # AC-13: restating a resolved item keeps it resolved; a later attempt omitting
        # it re-flags it unresolved, because ingest prunes keys omitted on restatement (D6).
        evidence, ref = _seeded_evidence()
        state = _state(evidence=evidence)

        first = _analytical(_obs("payments_rev", item="Payments segment", ref=ref))
        state.findings.ingest(first, state.evidence)
        assert gates_module.named_item_gate(first, state) is not None

        # Attempt 2 restates the item as resolved -> no longer chased.
        resolved = _analytical(
            _obs(
                "payments_rev",
                item="Payments segment",
                status="resolved",
                claim="$12M",
                ref=ref,
            )
        )
        state.findings.ingest(resolved, state.evidence)
        assert gates_module.named_item_gate(resolved, state) is None

        # Attempt 3 restates it again -> still resolved.
        state.findings.ingest(resolved, state.evidence)
        assert gates_module.named_item_gate(resolved, state) is None

        # Attempt 4 omits the aspect entirely -> ingest prunes the key, so the earlier
        # resolution no longer suppresses a fresh unresolved report of the same item.
        state.findings.ingest(_analytical(_obs("other", ref=ref)), state.evidence)
        reflagged = _analytical(_obs("payments_rev", item="Payments segment", ref=ref))
        assert gates_module.named_item_gate(reflagged, state) is not None

    def test_same_aspect_collapse_does_not_resurrect_a_resolved_item(self) -> None:
        # Plan §3's R3: two observations share one aspect within a single attempt, the
        # resolved one authored *before* the unresolved one. ingest keeps only the last
        # (2 in -> 1 out), so reading the projection alone would discard the resolution
        # and re-flag a settled item. The gate unions candidate + projection, so it
        # reads as resolved.
        evidence, ref = _seeded_evidence()
        state = _state(evidence=evidence)
        candidate = _analytical(
            _obs(
                "payments_rev",
                item="Payments segment",
                status="resolved",
                claim="$12M",
                ref=ref,
            ),
            _obs("payments_rev", item="Payments segment", ref=ref),
        )
        state.findings.ingest(candidate, state.evidence)

        # Pin the collapse this test rests on: 2 observations in, 1 out, the last surviving.
        projection = state.findings.projection()
        assert isinstance(projection, AnalyticalFindings)
        assert len(projection.observations) == 1
        surviving = projection.observations[0].named_item
        assert surviving is not None and surviving.status == "unresolved"

        # Projection alone says "unresolved"; the union with the candidate says resolved.
        assert unresolved_named_items(projection.observations) != {}
        assert gates_module.named_item_gate(candidate, state) is None
        assert state.named_item_rejections_total == 0


class TestMissingEntityGate:
    def test_fires_when_entity_never_searched(self) -> None:
        state = _state(
            expected_entities={"Acme", "Globex"},
            searched_entities={"Acme"},
            findings=_reported("Acme"),
        )
        candidate = AgentFindings(metric_requested="revenue", findings=())
        reason = gates_module.missing_entity_gate(candidate, state)
        assert reason is not None
        assert "search" in reason and "Globex" in reason

    def test_fires_when_searched_but_unreported(self) -> None:
        """The gap this change closes: an entity searched but omitted from the report
        used to slip past both this gate and synthesis' unsearched-stub backstop."""
        state = _state(
            expected_entities={"Acme"}, searched_entities={"Acme"}, findings=FindingsLedger()
        )
        candidate = AgentFindings(metric_requested="revenue", findings=())
        reason = gates_module.missing_entity_gate(candidate, state)
        assert reason is not None
        assert "report" in reason and "Acme" in reason

    def test_fires_when_reported_but_unsearched(self) -> None:
        """The dual hole stays closed: reporting available=false without searching does
        not wave an entity through — the must-retrieve guarantee is preserved."""
        state = _state(
            expected_entities={"Acme"}, searched_entities=set(), findings=_reported("Acme")
        )
        candidate = AgentFindings(metric_requested="revenue", findings=())
        reason = gates_module.missing_entity_gate(candidate, state)
        assert reason is not None
        assert "search" in reason and "Acme" in reason

    def test_silent_when_all_searched_and_reported(self) -> None:
        state = _state(
            expected_entities={"Acme"}, searched_entities={"Acme"}, findings=_reported("Acme")
        )
        candidate = AgentFindings(metric_requested="revenue", findings=())
        assert gates_module.missing_entity_gate(candidate, state) is None

    def test_ignores_analytical_candidates(self) -> None:
        state = _state(expected_entities={"Acme"}, searched_entities=set())
        candidate = AnalyticalFindings(question="q", observations=())
        assert gates_module.missing_entity_gate(candidate, state) is None

    def test_fires_regardless_of_retry_budget(self) -> None:
        """Unlike the sufficiency gate, this never waives on iteration/budget pressure."""
        state = _state(
            expected_entities={"Acme"},
            searched_entities=set(),
            iteration=3,  # last possible iteration
        )
        candidate = AgentFindings(metric_requested="revenue", findings=())
        assert gates_module.missing_entity_gate(candidate, state) is not None


class TestAnalyticalInsufficiencyGate:
    def _thin_candidate(self) -> AnalyticalFindings:
        return AnalyticalFindings(
            question="q",
            observations=(
                Observation(aspect="a", claim="x", evidence_chunks=[], confidence="high"),
            ),
        )

    def test_fires_when_retry_budget_remains(self) -> None:
        state = _state(iteration=0)
        reason = gates_module.analytical_insufficiency_gate(self._thin_candidate(), state)
        assert reason is not None

    def test_waives_on_last_iteration(self) -> None:
        state = _state(iteration=3)  # effort.max_iterations - 1
        assert gates_module.analytical_insufficiency_gate(self._thin_candidate(), state) is None

    def test_waives_when_rejection_cap_reached(self) -> None:
        state = _state(iteration=0, insufficiency_rejections=1)  # == max_insufficiency_rejections
        assert gates_module.analytical_insufficiency_gate(self._thin_candidate(), state) is None

    def test_ignores_extraction_candidates(self) -> None:
        state = _state(iteration=0)
        candidate = AgentFindings(metric_requested="revenue", findings=())
        assert gates_module.analytical_insufficiency_gate(candidate, state) is None


class TestContractC3:
    def test_structural_reason_wins_over_sufficiency(self) -> None:
        """A candidate that fails a coverage gate *and* a sufficiency gate gets the
        coverage reason — gates are ordered structural before sufficiency and the
        dispatcher stops at the first firing reason."""
        from src.services.chat.agent.tools import gates_for

        state = _state(expected_entities={"Acme"}, searched_entities=set(), iteration=0)
        # An AgentFindings candidate only ever runs the missing_entity_gate (the only
        # gate registered for report_findings) — confirm it's first/only and fires.
        registered = gates_for("report_findings")
        assert registered[0] is gates_module.missing_entity_gate

        candidate = AgentFindings(metric_requested="revenue", findings=())
        reason = next((r for g in registered if (r := g(candidate, state)) is not None), None)
        assert reason is not None and "Acme" in reason
