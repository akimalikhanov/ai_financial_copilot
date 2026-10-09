"""Size control for what goes onto Langfuse observations.

A GENERATION input is the prompt as sent, and an agent run sends nearly the same prompt
every turn: the transcript only grows, so turn N re-logs every excerpt turns 0..N-1
already logged. Within one trace, a message an earlier generation logged is replaced by a
pointer to it, and whatever is logged is capped: system prompts (versioned, named in the
trace metadata), excerpt bodies, and any one message.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
from collections.abc import Iterator, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from src.utils.config import (
    get_langfuse_trace_dedup_min_chars,
    get_langfuse_trace_excerpt_chars,
    get_langfuse_trace_max_hits,
    get_langfuse_trace_message_chars,
    get_langfuse_trace_system_prompt_chars,
)

_EXCERPT_RE = re.compile(r"(<retrieved_excerpt[^>]*>\n)(.*?)(\n</retrieved_excerpt>)", re.DOTALL)


@dataclass
class _DedupScope:
    # Message digest → the generation that first logged it.
    seen: dict[str, str] = field(default_factory=dict)
    generations: int = 0


_scope: ContextVar[_DedupScope | None] = ContextVar("lf_trace_dedup_scope", default=None)


@contextlib.contextmanager
def dedup_scope() -> Iterator[None]:
    """Generations started inside log each repeated message once. Open one per trace."""
    token = _scope.set(_DedupScope())
    try:
        yield
    finally:
        _scope.reset(token)


def cap_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}… [+{len(text) - max_chars} chars]"


def cap_excerpts(text: str, max_chars: int) -> str:
    """Trim each `<retrieved_excerpt>` body, keeping its tag (ref id, source doc)."""

    def _trim(m: re.Match[str]) -> str:
        return m.group(1) + cap_text(m.group(2), max_chars) + m.group(3)

    return _EXCERPT_RE.sub(_trim, text)


def cap_list(items: Sequence[Any], max_items: int | None = None) -> list[Any]:
    """The first entries of a list; callers log the full length beside it."""
    limit = get_langfuse_trace_max_hits() if max_items is None else max_items
    return list(items[:limit])


def _cap_content(role: str, content: Any) -> Any:
    def _one(text: str) -> str:
        if role == "system":
            return cap_text(text, get_langfuse_trace_system_prompt_chars())
        return cap_text(
            cap_excerpts(text, get_langfuse_trace_excerpt_chars()),
            get_langfuse_trace_message_chars(),
        )

    if isinstance(content, str):
        return _one(content)
    if isinstance(content, list):
        # Image parts stay: Langfuse's media manager swaps their data URIs for references.
        return [
            {**p, "text": _one(p["text"])}
            if isinstance(p, dict) and isinstance(p.get("text"), str)
            else p
            for p in content
        ]
    return content


def _content_len(content: Any) -> int:
    if isinstance(content, str):
        return len(content)
    return len(json.dumps(content, default=str))


def compact_messages(messages: Sequence[dict[str, Any]], generation_name: str) -> list[dict]:
    """Serialized chat messages, capped and (inside a `dedup_scope`) de-duplicated."""
    scope = _scope.get()
    label = generation_name
    if scope is not None:
        scope.generations += 1
        label = f"{generation_name} #{scope.generations}"
    min_dedup = get_langfuse_trace_dedup_min_chars()

    out: list[dict] = []
    for msg in messages:
        role = str(msg.get("role", ""))
        original = msg.get("content")
        if scope is not None and _content_len(original) >= min_dedup:
            digest = hashlib.sha1(
                json.dumps(
                    [role, original, msg.get("tool_call_id"), msg.get("tool_calls")],
                    default=str,
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            first = scope.seen.get(digest)
            if first is not None:
                out.append(
                    {
                        **msg,
                        "content": f"[logged in {first}; {_content_len(original)} chars]",
                    }
                )
                continue
            scope.seen[digest] = label
        out.append({**msg, "content": _cap_content(role, original)})
    return out


def trace_params(params: dict[str, Any]) -> dict[str, Any]:
    """Call parameters as Langfuse `model_parameters` (scalars only)."""
    out: dict[str, Any] = {}
    for k, v in params.items():
        if v is None or k == "tools":
            continue
        if isinstance(v, str | int | float | bool):
            out[k] = v
        elif k == "response_format" and isinstance(v, dict):
            schema = v.get("json_schema")
            name = schema.get("name") if isinstance(schema, dict) else None
            out[k] = f"{v.get('type')}:{name}" if name else str(v.get("type"))
        elif k == "tool_choice" and isinstance(v, dict):
            out[k] = json.dumps(v)
    return out
