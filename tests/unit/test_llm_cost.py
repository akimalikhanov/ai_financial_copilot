"""OpenAI cost: cached reads and cache writes are both subsets of input_tokens."""

from __future__ import annotations

import pytest

from src.services.llm_adapters.base_adapter import LLMResponseStats
from src.utils.llm_utils import calc_cost_openai

# Per 1K tokens, so 1,000 tokens of a kind cost exactly its rate.
PRICING = {"input": 1.0, "cached_input": 0.1, "output": 10.0}


@pytest.mark.parametrize(
    ("cached", "written", "write_price", "expected"),
    [
        (0, 0, None, 3.0),  # 3K plain input
        (1000, 0, None, 2.1),  # 2K plain + 1K read at 0.1
        (0, 1000, None, 3.0),  # no surcharge: writes billed as plain input
        (0, 1000, 1.25, 3.25),  # gpt-5.6+: writes at 1.25x
        (1000, 1000, 1.25, 2.35),  # 1K plain + 1K read + 1K written
        (3000, 1000, 1.25, 0.3),  # writes clamped to what reads leave
    ],
)
def test_cache_write_pricing(
    cached: int, written: int, write_price: float | None, expected: float
) -> None:
    stats = LLMResponseStats(
        input_tokens=3000, cached_input_tokens=cached, cache_write_tokens=written, output_tokens=0
    )
    pricing = {**PRICING, "cache_write_input": write_price}
    assert calc_cost_openai(stats, pricing) == pytest.approx(expected)


def test_output_includes_reasoning() -> None:
    stats = LLMResponseStats(input_tokens=0, output_tokens=1000, reasoning_tokens=800)
    assert calc_cost_openai(stats, PRICING) == pytest.approx(10.0)
