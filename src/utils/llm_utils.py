from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from src.utils.config import load_models_config


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def elapsed_ms(start_ms: float, end_ms: float | None = None) -> float:
    if end_ms is None:
        end_ms = now_ms()
    return max(0.0, end_ms - start_ms)


def compute_tps(
    *,
    output_tokens: int | None,
    latency_ms: float | None,
    ttft_ms: float | None = None,
) -> float | None:
    if output_tokens is None or latency_ms is None:
        return None
    if output_tokens <= 0:
        return 0.0
    effective_latency_ms = latency_ms
    if ttft_ms is not None:
        effective_latency_ms = max(0.0, latency_ms - ttft_ms)
    if effective_latency_ms <= 0.0:
        return None
    return output_tokens / (effective_latency_ms / 1000.0)


def get_pricing_for_model(
    provider: str,
    model_name: str,
    config_path: str | Path | None = None,
) -> dict[str, Any] | None:
    """
    Get pricing information for a model by provider and model_name.

    Args:
        provider: Provider name (e.g., "openai", "google", "vllm")
        model_name: Model name (e.g., "gpt-5.2", "gemini-3-flash-preview")
        config_path: Optional path to models.yaml

    Returns:
        Pricing dict with keys: input, output, cached_input, cached_output, etc.
        Returns None if model not found or pricing not available.
    """
    config = load_models_config(config_path)
    models = config.get("models", [])

    for model in models:
        if model.get("provider") == provider and model.get("model_name") == model_name:
            pricing = model.get("pricing")
            if pricing is None:
                return None
            # Handle "null" string or None values
            pricing = {k: (None if v in ("null", None) else v) for k, v in pricing.items()}
            return pricing

    return None


def _calc_llm_cost_shared(
    stats: Any,
    pricing: dict[str, Any],
    *,
    reasoning_in_output: bool,
    implicit_cache_discount: float | None = None,
    tokens_per_unit: int = 1000,  # keep 1000 if your pricing is "per 1K tokens"
) -> float | None:
    """
    Shared token cost calculator.

    Args:
        stats: object with fields: input_tokens, output_tokens, reasoning_tokens, cached_input_tokens
        pricing: dict with keys: input, output, optional cached_input
        reasoning_in_output:
            - True  => reasoning already included in output_tokens (OpenAI)
            - False => reasoning separate, must add (Gemini)
        implicit_cache_discount:
            - None => use pricing["cached_input"] if present, else treat cached_input cost as 0
            - e.g. 0.10 => cached input billed at 10% of normal input price (Gemini implicit caching)
        tokens_per_unit: 1000 for "per 1K tokens", 1_000_000 for "per 1M tokens"

    Returns:
        Total cost (float) or None if pricing is missing required keys.
    """
    if not pricing:
        return None

    # Require at least input + output prices
    if pricing.get("input") is None or pricing.get("output") is None:
        return None

    pin = float(pricing["input"])
    pout = float(pricing["output"])

    in_tokens = int(getattr(stats, "input_tokens", 0) or 0)
    out_tokens = int(getattr(stats, "output_tokens", 0) or 0)
    reasoning = int(getattr(stats, "reasoning_tokens", 0) or 0)
    cached_in = int(getattr(stats, "cached_input_tokens", 0) or 0)

    # Clamp cached input to sane range
    cached_in = max(min(cached_in, in_tokens), 0)

    billable_in = max(in_tokens - cached_in, 0)
    billable_cached_in = cached_in

    billable_out = out_tokens if reasoning_in_output else (out_tokens + reasoning)

    # Cached input price logic:
    # 1) if explicit cached_input price provided => use it
    # 2) else if implicit discount exists => pin * discount
    # 3) else => 0
    if pricing.get("cached_input") is not None:
        pcached = float(pricing["cached_input"])
    elif implicit_cache_discount is not None:
        pcached = pin * float(implicit_cache_discount)
    else:
        pcached = 0.0

    denom = float(tokens_per_unit)

    cost = (
        (billable_in / denom) * pin
        + (billable_cached_in / denom) * pcached
        + (billable_out / denom) * pout
    )
    return cost


def calc_cost_openai(stats: Any, pricing: dict[str, Any]) -> float | None:
    """
    OpenAI rules:
      - reasoning_tokens already included in output_tokens
      - cached_input_tokens billed at cached_input price (if provided)
    """
    return _calc_llm_cost_shared(
        stats,
        pricing,
        reasoning_in_output=True,
        implicit_cache_discount=None,
        tokens_per_unit=1000,
    )


def calc_cost_google(stats: Any, pricing: dict[str, Any]) -> float | None:
    """
    Gemini rules (implicit caching default):
      - reasoning_tokens are separate => add them to output_tokens
      - cached input billed at ~10% of input if cached_input price not provided
    """
    return _calc_llm_cost_shared(
        stats,
        pricing,
        reasoning_in_output=False,
        implicit_cache_discount=0.10,
        tokens_per_unit=1000,
    )
