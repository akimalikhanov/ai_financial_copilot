"""Router carry-over: prior findings reach the router as a compact digest."""

from __future__ import annotations

from src.schemas.query_router import RouterInput
from src.services.router.router import _build_messages, _digest_findings_block

FINDINGS = """[FINDINGS]
Target currency: EUR | Operation: argmin

1. Atreca, Inc. [high confidence] Atreca's net loss was $97.2M. | evidence: S1, S8
   - net income (FY2022 / 2022-12-31): USD -97,157.0K
2. Datalogic [high confidence] Datalogic's net income was $32.0M. | evidence: S3
   - net income (FY2022 / 2022-12-31): EUR 30,126.0K | from USD 32,000.0K | rate: 0.9414 | ⚠ UNVERIFIED: value not located in cited excerpt
3. NuCana plc [not disclosed] The filings do not report net income.

Conclusion: cost inflation drove the compression
Unresolved: Not resolved: Globex
[END FINDINGS]"""


def test_digest_keeps_values_drops_refs() -> None:
    d = _digest_findings_block(FINDINGS)

    assert "- net income (FY2022 / 2022-12-31): USD -97,157.0K" in d
    assert "- net income (FY2022 / 2022-12-31): EUR 30,126.0K" in d
    assert "3. NuCana plc [not disclosed] The filings do not report net income." in d
    assert "Conclusion: cost inflation drove the compression" in d
    assert "Unresolved: Not resolved: Globex" in d
    # Routing-irrelevant detail is dropped.
    assert "evidence:" not in d
    assert "rate:" not in d
    assert "UNVERIFIED" not in d
    # Block markers are not signal.
    assert "[FINDINGS]" not in d


def test_digest_is_capped() -> None:
    huge = "[FINDINGS]\n" + "\n".join(f"   - revenue (FY{i}): USD 1.0M" for i in range(500))
    assert len(_digest_findings_block(huge, max_chars=200)) <= 200


def test_block_reaches_the_router_prompt() -> None:
    msgs = _build_messages(
        RouterInput(query="summarize as a table", prior_findings_block=FINDINGS),
        system="SYS",
    )
    user = msgs[-1].content or ""

    assert "Data already retrieved in this conversation" in user
    assert "- net income (FY2022 / 2022-12-31): USD -97,157.0K" in user
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
