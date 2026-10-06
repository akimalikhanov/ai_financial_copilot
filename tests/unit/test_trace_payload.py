"""What the app sends to Langfuse: capped, de-duplicated, errors marked."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Iterator
from typing import Any

import pytest

from src.observability import langfuse as lf_client
from src.observability import trace_payload as tp
from src.services.llm_adapters.base_adapter import ChatMessage, Role
from src.services.retrieval.context_assembler import wrap_excerpt


@pytest.fixture(autouse=True)
def _caps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_TRACE_SYSTEM_PROMPT_CHARS", "10")
    monkeypatch.setenv("LANGFUSE_TRACE_EXCERPT_CHARS", "20")
    monkeypatch.setenv("LANGFUSE_TRACE_MESSAGE_CHARS", "1000")
    monkeypatch.setenv("LANGFUSE_TRACE_DEDUP_MIN_CHARS", "50")
    monkeypatch.setenv("LANGFUSE_TRACE_MAX_HITS", "3")


def _tool_result(n: int) -> dict[str, Any]:
    body = "\n\n".join(wrap_excerpt(f"S{i}", "doc.pdf", False, "x" * 200) for i in range(n))
    return {"role": "tool", "tool_call_id": f"call{n}", "content": body}


def test_excerpt_bodies_are_capped_and_tags_kept() -> None:
    [msg] = tp.compact_messages([_tool_result(2)], "gen")
    assert msg["content"].count("<retrieved_excerpt") == 2
    assert msg["content"].count("</retrieved_excerpt>") == 2
    assert 'id="S1" source_doc="doc.pdf"' in msg["content"]
    assert "x" * 20 + "… [+180 chars]" in msg["content"]
    assert "x" * 21 not in msg["content"]


def test_system_prompt_is_capped() -> None:
    [msg] = tp.compact_messages([{"role": "system", "content": "s" * 100}], "gen")
    assert msg["content"] == "s" * 10 + "… [+90 chars]"


def test_message_cap_bounds_text_outside_excerpts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_TRACE_MESSAGE_CHARS", "30")
    [msg] = tp.compact_messages([{"role": "user", "content": "u" * 100}], "gen")
    assert msg["content"] == "u" * 30 + "… [+70 chars]"


def test_repeated_messages_are_logged_once_per_trace() -> None:
    system = {"role": "system", "content": "s" * 100}
    turn0 = [system, {"role": "user", "content": "q"}]
    turn1 = [*turn0, _tool_result(1)]
    with tp.dedup_scope():
        first = tp.compact_messages(turn0, "llm.complete_with_tools")
        second = tp.compact_messages(turn1, "llm.complete_with_tools")

    assert first[0]["content"].startswith("s" * 10)
    # Seen in generation #1: a pointer, not the text.
    assert second[0]["content"] == "[logged in llm.complete_with_tools #1; 100 chars]"
    # Short messages are always kept — a pointer would be no smaller.
    assert second[1]["content"] == "q"
    # New this turn: logged (capped).
    assert "<retrieved_excerpt" in second[2]["content"]
    assert second[2]["tool_call_id"] == "call1"


def test_no_dedup_outside_a_scope() -> None:
    msgs = [{"role": "system", "content": "s" * 100}]
    tp.compact_messages(msgs, "gen")
    [again] = tp.compact_messages(msgs, "gen")
    assert again["content"].startswith("s" * 10)


def test_scopes_do_not_share_state() -> None:
    msgs = [{"role": "system", "content": "s" * 100}]
    with tp.dedup_scope():
        tp.compact_messages(msgs, "gen")
    with tp.dedup_scope():
        [msg] = tp.compact_messages(msgs, "gen")
    assert not msg["content"].startswith("[logged in")


def test_image_parts_are_left_for_the_media_manager() -> None:
    content = [
        {"type": "text", "text": "t" * 2000},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    [msg] = tp.compact_messages([{"role": "user", "content": content}], "gen")
    assert msg["content"][1] == content[1]
    assert len(msg["content"][0]["text"]) < 2000


def test_cap_list_uses_the_configured_limit() -> None:
    assert tp.cap_list(list(range(10))) == [0, 1, 2]


def test_trace_params_keeps_scalars_and_names_the_schema() -> None:
    params = {
        "temperature": 0.0,
        "max_tokens": 300,
        "stop": None,
        "tools": [{"type": "function"}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "router"}},
        "tool_choice": {"type": "function", "function": {"name": "x"}},
    }
    assert tp.trace_params(params) == {
        "temperature": 0.0,
        "max_tokens": 300,
        "response_format": "json_schema:router",
        "tool_choice": '{"type": "function", "function": {"name": "x"}}',
    }


class _Obs:
    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)


class _Client:
    def __init__(self) -> None:
        self.observations: list[_Obs] = []

    @contextlib.contextmanager
    def start_as_current_observation(self, **_kwargs: Any) -> Iterator[_Obs]:
        obs = _Obs()
        self.observations.append(obs)
        yield obs


def test_span_marks_error_when_the_block_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client()
    monkeypatch.setattr(lf_client, "get_client", lambda: client)
    with pytest.raises(ValueError), lf_client.span("s"):
        raise ValueError("boom")
    assert client.observations[0].updates == [
        {"level": "ERROR", "status_message": "ValueError: boom"}
    ]


def test_span_marks_cancellation(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client()
    monkeypatch.setattr(lf_client, "get_client", lambda: client)
    with pytest.raises(asyncio.CancelledError), lf_client.span("s"):
        raise asyncio.CancelledError
    assert client.observations[0].updates[0]["level"] == "ERROR"
    assert "cancelled" in client.observations[0].updates[0]["status_message"]


@pytest.mark.asyncio
async def test_generation_is_marked_when_the_call_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.services import llm_router

    client = _Client()
    monkeypatch.setattr(llm_router._lf_mod, "get_client", lambda: client)

    class _Adapter:
        async def complete(self, **_kwargs: Any) -> Any:
            raise TimeoutError("slow")

    routed = llm_router.RoutedLLM(
        adapter=_Adapter(),  # type: ignore[arg-type]
        provider="fake",
        model_id="m",
        default_params={},
        default_stream=False,
        capabilities={},
    )
    with pytest.raises(TimeoutError):
        await routed.complete([ChatMessage(role=Role.user, content="hi")])
    assert client.observations[0].updates == [
        {"level": "ERROR", "status_message": "TimeoutError: slow"}
    ]
