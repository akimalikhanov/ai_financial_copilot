from __future__ import annotations

import json

from pydantic import ValidationError

from src.schemas.query_router import RouterOutput


def parse_router_response(text: str) -> tuple[RouterOutput | None, str | None]:
    """Returns (RouterOutput, None) on success, (None, error_msg) on failure."""
    text = (text or "").strip()
    if not text:
        return None, "Empty response"
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}"
    try:
        out = RouterOutput.model_validate(data)
    except ValidationError as e:
        return None, f"Schema validation failed: {e}"

    # Clamp free-text fields to avoid bloated logs/state
    out.reasoning = out.reasoning[:1000]
    out.user_intent = out.user_intent[:300]
    return out, None
