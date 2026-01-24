from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Optional

import yaml

from src.services.llm_adapters.base_adapter import LLMResponseStats


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def elapsed_ms(start_ms: float, end_ms: Optional[float] = None) -> float:
    if end_ms is None:
        end_ms = now_ms()
    return max(0.0, end_ms - start_ms)


def compute_tps(
    *,
    output_tokens: Optional[int],
    latency_ms: Optional[float],
    ttft_ms: Optional[float] = None,
) -> Optional[float]:
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


def _expand_env_vars(value: Any) -> Any:
    """Recursively expand ${VAR:-default} patterns in YAML values."""
    if isinstance(value, str):
        # Match ${VAR:-default} pattern
        pattern = r'\$\{([^:}]+)(?::-([^}]*))?\}'
        
        def replacer(match: re.Match[str]) -> str:
            var_name = match.group(1)
            default = match.group(2) if match.group(2) is not None else ""
            return os.getenv(var_name, default)
        
        return re.sub(pattern, replacer, value)
    elif isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    else:
        return value


def load_models_config(config_path: Optional[str | Path] = None) -> dict[str, Any]:
    """
    Load models.yaml config file with environment variable expansion.
    
    Args:
        config_path: Path to models.yaml. If None, uses infra/config/models.yaml relative to project root.
    
    Returns:
        Parsed YAML dict with env vars expanded.
    """
    if config_path is None:
        # Assume we're in src/utils/, go up to project root
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "infra" / "config" / "models.yaml"
    else:
        config_path = Path(config_path)
    
    with open(config_path, "r") as f:
        raw_data = yaml.safe_load(f)
    
    # Expand environment variables
    return _expand_env_vars(raw_data)


def get_pricing_for_model(
    provider: str,
    model_name: str,
    config_path: Optional[str | Path] = None,
) -> Optional[dict[str, Any]]:
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
        if (
            model.get("provider") == provider
            and model.get("model_name") == model_name
        ):
            pricing = model.get("pricing")
            if pricing is None:
                return None
            # Handle "null" string or None values
            pricing = {k: (None if v in ("null", None) else v) for k, v in pricing.items()}
            return pricing
    
    return None


def calc_cost_openai(stats: LLMResponseStats, pricing: dict[str, Any]) -> Optional[float]:
    """
    Calculate cost for OpenAI models.
    
    OpenAI rule:
      - reasoning_tokens are INCLUDED in output_tokens (do NOT add them)
      - cached_input_tokens (if any) are billed at cached_input price
    
    Args:
        stats: LLMResponseStats with token counts
        pricing: Pricing dict with keys: input, output, cached_input
    
    Returns:
        Cost in USD, or None if pricing is incomplete
    """
    if not pricing:
        return None
    
    in_tokens = int(stats.input_tokens or 0)
    out_tokens = int(stats.output_tokens or 0)
    cached_in = int(stats.cached_input_tokens or 0)

    billable_in = max(in_tokens - cached_in, 0)
    billable_cached_in = cached_in
    billable_out = out_tokens  # ✅ reasoning already included

    pin = float(pricing.get("input") or 0.0)
    pout = float(pricing.get("output") or 0.0)
    pcached = float(pricing.get("cached_input") or 0.0)

    cost = (billable_in / 1000) * pin \
         + (billable_cached_in / 1000) * pcached \
         + (billable_out / 1000) * pout

    return cost


def calc_cost_google(stats: LLMResponseStats, pricing: dict[str, Any]) -> Optional[float]:
    """
    Calculate cost for Google (Gemini) models.
    
    Google (Gemini) rule (implicit caching only):
      - reasoning_tokens are SEPARATE => add them to output_tokens
      - cached_input_tokens are not charged separately for implicit caching
        (treat cached_input_price as 0 unless you explicitly enable caching)
    
    Args:
        stats: LLMResponseStats with token counts
        pricing: Pricing dict with keys: input, output
    
    Returns:
        Cost in USD, or None if pricing is incomplete
    """
    if not pricing:
        return None
    
    in_tokens = int(stats.input_tokens or 0)
    out_tokens = int(stats.output_tokens or 0)
    reasoning = int(stats.reasoning_tokens or 0)

    billable_in = in_tokens
    billable_out = out_tokens + reasoning  # ✅ must add thinking/reasoning

    pin = float(pricing.get("input") or 0.0)
    pout = float(pricing.get("output") or 0.0)

    cost = (billable_in / 1000) * pin \
         + (billable_out / 1000) * pout

    return cost
