"""Router carry-over: prior findings reach the router as a compact digest."""

from __future__ import annotations

from src.schemas.query_router import RouterInput
from src.services.router.router import _build_messages, _digest_findings_block

FINDINGS = """[STRUCTURED FINDINGS]
Metric: net income | Operation: argmin

Atreca, Inc.           | USD -97,157.0K | native | period: 2022-12-31 | chunks: S1, S8
Datalogic              | EUR 30,126.0K  | from USD 32,000.0K | rate: 0.9414 | period: 2022-12-31 | chunks: S3
NuCana plc             | N/A | not available: not found in retrieved context
[END STRUCTURED FINDINGS]"""

OBSERVATIONS = """[AGENT OBSERVATIONS]
Question: why did margin compress?

1. [high confidence] Input costs rose 12% | evidence: S1, S2 | refuted_by:
2. [not disclosed] FX impact on gross margin
Conclusion: cost inflation drove the compression
Unresolved (do not assert as fact): no FX quantification found
[END AGENT OBSERVATIONS]"""


def test_digest_keeps_values_drops_refs() -> None:
    d = _digest_findings_block(FINDINGS)

    assert "Atreca, Inc. | USD -97,157.0K" in d
    assert "Datalogic | EUR 30,126.0K" in d
    assert "NuCana plc | N/A | not available" in d
    # Routing-irrelevant detail is dropped.
    assert "chunks:" not in d
    assert "rate:" not in d
    assert "native" not in d
    # Block markers are not signal.
    assert "[STRUCTURED FINDINGS]" not in d


def test_digest_handles_observations() -> None:
    d = _digest_findings_block(OBSERVATIONS)

    assert "Input costs rose 12%" in d
    assert "[not disclosed] FX impact on gross margin" in d
    assert "Conclusion: cost inflation drove the compression" in d
    assert "evidence:" not in d
    assert "Question:" not in d


def test_digest_is_capped() -> None:
    huge = "[STRUCTURED FINDINGS]\n" + "\n".join(f"Entity{i} | USD 1.0M" for i in range(500))
    assert len(_digest_findings_block(huge, max_chars=200)) <= 200


def test_block_reaches_the_router_prompt() -> None:
    msgs = _build_messages(
        RouterInput(query="summarize as a table", prior_findings_block=FINDINGS),
        system="SYS",
    )
    user = msgs[-1].content or ""

    assert "Data already retrieved in this conversation" in user
    assert "Atreca, Inc. | USD -97,157.0K" in user
    assert "User query: summarize as a table" in user


def test_router_prompt_documents_the_carryover_block() -> None:
    """The prompt's heading must match the one _build_messages emits, or the guidance
    describes a block the router never sees."""
    from src.services.prompts.prompt_loader import get_prompt_loader
    from src.utils.config import get_query_router_prompt_version

    template = get_prompt_loader().load("query_router", get_query_router_prompt_version()).template
    rendered = (
        _build_messages(RouterInput(query="q", prior_findings_block=FINDINGS), system="SYS")[
            -1
        ].content
        or ""
    )

    heading = "Data already retrieved in this conversation"
    assert heading in template
    assert heading in rendered
    # The D4 minimal pair: same surface form, opposite routes.
    assert "convert that to USD at 1.07" in template
    assert "no user-supplied rate" in template


def test_absent_block_adds_nothing() -> None:
    msgs = _build_messages(RouterInput(query="what was revenue?"), system="SYS")
    user = msgs[-1].content or ""

    assert "Data already retrieved" not in user
    assert user == "User query: what was revenue?"
