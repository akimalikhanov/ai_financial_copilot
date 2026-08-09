"""Staleness rules for a carried findings block.

A block is dropped after FOLLOWUP_MAX_INHERIT_HOPS inherited turns, or when the resolved
document scope moves. Either way the router sees no carried data and the follow-up
re-retrieves rather than restating numbers that no longer describe what was asked.
"""

from __future__ import annotations

import pytest

from src.schemas import chat as schemas
from src.services.chat.tasks import _latest_findings_block, _scope_moved

BLOCK = "[STRUCTURED FINDINGS]\nAcme Corp | USD 1.0M\n[END STRUCTURED FINDINGS]"
DOCS = ["doc-a", "doc-b"]


def _history(hops: int, doc_ids: list[str] | None = None) -> list[schemas.ChatMessage]:
    return [
        schemas.ChatMessage(role=schemas.Role.user, content="q"),
        schemas.ChatMessage(
            role=schemas.Role.assistant,
            content="a",
            findings_block=BLOCK,
            findings_block_hops=hops,
            findings_block_doc_ids=doc_ids,
        ),
    ]


@pytest.mark.parametrize(("hops", "expected"), [(0, 1), (1, 2), (2, 3)])
def test_block_carried_under_cap(hops: int, expected: int) -> None:
    carried = _latest_findings_block(_history(hops))
    assert carried.block == BLOCK
    assert carried.hops == expected
    assert carried.outcome == "carried"


def test_block_dropped_at_cap() -> None:
    # 3 hops already inherited: a 4th would exceed the default cap.
    carried = _latest_findings_block(_history(3))
    assert carried.block is None
    assert carried.outcome == "dropped_hop_cap"


def test_cap_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOLLOWUP_MAX_INHERIT_HOPS", "1")
    assert _latest_findings_block(_history(0)).block == BLOCK
    assert _latest_findings_block(_history(1)).outcome == "dropped_hop_cap"


def test_no_block_and_freshly_retrieved() -> None:
    assert _latest_findings_block(None).outcome == "none"
    assert _latest_findings_block([]).outcome == "none"
    plain = [schemas.ChatMessage(role=schemas.Role.assistant, content="a")]
    assert _latest_findings_block(plain).outcome == "none"


def test_most_recent_block_wins() -> None:
    older = schemas.ChatMessage(
        role=schemas.Role.assistant, content="a", findings_block="OLD", findings_block_hops=0
    )
    newer = schemas.ChatMessage(
        role=schemas.Role.assistant, content="b", findings_block=BLOCK, findings_block_hops=1
    )
    carried = _latest_findings_block([older, newer])
    assert (carried.block, carried.hops) == (BLOCK, 2)


# --- scope invalidation ---


def test_scope_unchanged_carries() -> None:
    carried = _latest_findings_block(_history(0, DOCS), list(DOCS), check_scope=True)
    assert carried.block == BLOCK
    assert carried.doc_ids == DOCS


def test_scope_order_does_not_matter() -> None:
    carried = _latest_findings_block(_history(0, DOCS), ["doc-b", "doc-a"], check_scope=True)
    assert carried.outcome == "carried"


def test_scope_narrowed_drops_block() -> None:
    carried = _latest_findings_block(_history(0, DOCS), ["doc-a"], check_scope=True)
    assert carried.block is None
    assert carried.outcome == "dropped_scope"


def test_all_docs_to_narrowed_drops_block() -> None:
    """None means "all documents" — narrowing away from it is a real scope change."""
    carried = _latest_findings_block(_history(0, None), ["doc-a"], check_scope=True)
    assert carried.outcome == "dropped_scope"


def test_all_docs_to_all_docs_carries() -> None:
    carried = _latest_findings_block(_history(0, None), None, check_scope=True)
    assert carried.outcome == "carried"


def test_scope_not_checked_for_the_router() -> None:
    """The router runs before scope resolves, so it must not drop on a scope mismatch."""
    carried = _latest_findings_block(_history(0, DOCS), None)
    assert carried.block == BLOCK


def test_hop_cap_takes_precedence_over_scope() -> None:
    carried = _latest_findings_block(_history(3, DOCS), ["doc-a"], check_scope=True)
    assert carried.outcome == "dropped_hop_cap"


@pytest.mark.parametrize(
    ("before", "now", "moved"),
    [
        (None, None, False),
        (None, ["a"], True),
        (["a"], None, True),
        (["a"], ["a"], False),
        (["a", "b"], ["b", "a"], False),
        (["a"], ["a", "b"], True),
        ([], [], False),
        ([], None, True),
    ],
)
def test_scope_moved(before: list[str] | None, now: list[str] | None, moved: bool) -> None:
    assert _scope_moved(before, now) is moved
