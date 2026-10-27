"""Unit tests for the Responses adapter's wire payload and result parsing.

Asserts on the JSON that leaves the process (respx on /v1/responses), like
test_openai_adapter.py, so a change to how the request is assembled can't pass these while
breaking the provider call.
"""

from __future__ import annotations

import base64
import json

import httpx
import pytest
import respx

from src.observability.metrics import LLM_DEGRADED_RETRIES
from src.services.llm_adapters import openai_responses_adapter as mod
from src.services.llm_adapters.base_adapter import (
    ChatMessage,
    ImagePart,
    OpaqueItem,
    Role,
    ToolCallRef,
)
from src.services.llm_adapters.openai_responses_adapter import OpenAIResponsesAdapter
from src.services.llm_runtime.exceptions import (
    LLMConnectionError,
    LLMInvalidRequestError,
    LLMRateLimitError,
    LLMReasoningReplayError,
    LLMServerError,
)

BASE_URL = "http://openai-test/v1"
URL = f"{BASE_URL}/responses"
PNG = b"\x89PNG\r\n\x1a\n" + b"payload"
PNG_B64 = base64.b64encode(PNG).decode("ascii")
SEARCH = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search documents",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
    },
}
REASONING = {
    "type": "reasoning",
    "id": "rs_1",
    "summary": [{"type": "summary_text", "text": "Need the segment table."}],
    "encrypted_content": "enc-blob",
}
CALL = {
    "type": "function_call",
    "id": "fc_1",
    "call_id": "call_1",
    "name": "search",
    "arguments": '{"q":"x"}',
    "status": "completed",
}
USAGE = {
    "input_tokens": 100,
    "input_tokens_details": {"cached_tokens": 40, "cache_write_tokens": 50},
    "output_tokens": 20,
    "output_tokens_details": {"reasoning_tokens": 12},
    "total_tokens": 120,
}


@pytest.fixture(autouse=True)
def _reset_summary_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "_summary_unavailable", False)


def _adapter(model: str = "gpt-5-mini", reasoning: bool = True) -> OpenAIResponsesAdapter:
    return OpenAIResponsesAdapter(
        default_model=model,
        capabilities={"reasoning": reasoning},
        api_key="k",
        base_url=BASE_URL,
    )


def _message(text: str = "ok", part_type: str = "output_text") -> dict:
    part: dict = {"type": part_type, ("refusal" if part_type == "refusal" else "text"): text}
    if part_type == "output_text":
        part["annotations"] = []
    return {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [part],
    }


def _response(*output: dict, status: str = "completed", incomplete: str | None = None) -> dict:
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "model": "gpt-5-mini-2025-08-07",
        "status": status,
        "incomplete_details": {"reason": incomplete} if incomplete else None,
        "output": list(output) or [_message()],
        "reasoning": {"effort": "medium", "context": "current_turn"},
        "service_tier": "default",
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": USAGE,
    }


def _bad_request(code: str | None = None, param: str | None = None, msg: str = "bad"):
    return httpx.Response(
        400,
        json={
            "error": {"message": msg, "type": "invalid_request_error", "code": code, "param": param}
        },
    )


async def _call(method: str, *args, adapter: OpenAIResponsesAdapter | None = None, **kw):
    adapter = adapter or _adapter()
    try:
        return await getattr(adapter, method)(*args, **kw)
    finally:
        await adapter.close()


def _sent(route, i: int = -1) -> dict:
    return json.loads(route.calls[i].request.content)


USER = [ChatMessage(role=Role.user, content="q")]


class TestRequestShape:
    @respx.mock
    async def test_items_and_invariants(self) -> None:
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=_response()))
        await _call(
            "complete",
            [
                ChatMessage(role=Role.system, content="sys"),
                ChatMessage(role=Role.user, content="hi"),
                ChatMessage(role=Role.assistant, content="earlier answer"),
                ChatMessage(role=Role.system, content="later"),
            ],
        )
        sent = _sent(route)
        assert sent["instructions"] == "sys"
        assert sent["input"] == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "earlier answer"},
            {"role": "developer", "content": "later"},
        ]
        assert sent["store"] is False and sent["truncation"] == "disabled"
        assert "include" not in sent and "stream" not in sent

    @respx.mock
    async def test_tool_call_pairing_by_call_id(self) -> None:
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=_response(CALL)))
        await _call(
            "complete_with_tools",
            [
                *USER,
                ChatMessage(
                    role=Role.assistant,
                    tool_calls=(ToolCallRef(id="call_0", name="search", arguments="{}"),),
                ),
                ChatMessage(role=Role.tool, content="result", tool_call_id="call_0"),
            ],
            [SEARCH],
        )
        sent = _sent(route)
        assert sent["input"][1:] == [
            {"type": "function_call", "call_id": "call_0", "name": "search", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_0", "output": "result"},
        ]
        assert sent["include"] == ["reasoning.encrypted_content"]
        assert sent["parallel_tool_calls"] is True and sent["store"] is False


class TestParameters:
    @staticmethod
    async def _sent(reasoning: bool, **kw) -> dict:
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=_response()))
        await _call("complete", USER, adapter=_adapter(reasoning=reasoning), **kw)
        return _sent(route)

    @respx.mock
    async def test_reasoning_model(self) -> None:
        sent = await self._sent(
            True,
            temperature=0.0,
            max_tokens=50,
            reasoning_effort="medium",
            reasoning_summary="auto",
            reasoning_context="",
            verbosity="low",
            enable_thinking=False,
            prompt_cache_key="router:v3",
        )
        assert "temperature" not in sent and "enable_thinking" not in sent
        assert sent["max_output_tokens"] == 50
        assert sent["reasoning"] == {"effort": "medium", "summary": "auto"}
        assert sent["text"] == {"verbosity": "low"}
        assert sent["prompt_cache_key"] == "router:v3"

    @respx.mock
    async def test_non_reasoning_model(self) -> None:
        sent = await self._sent(False, temperature=0.0, reasoning_effort="low", verbosity="low")
        assert sent["temperature"] == 0.0
        assert "reasoning" not in sent and "text" not in sent

    @respx.mock
    async def test_prompt_cache_key_dropped_when_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_PROMPT_CACHE_KEY_ENABLED", "false")
        assert "prompt_cache_key" not in await self._sent(True, prompt_cache_key="k")

    @respx.mock
    async def test_json_schema_response_format_is_flattened(self) -> None:
        schema = {"type": "object", "properties": {}}
        sent = await self._sent(
            False,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "route", "schema": schema, "strict": True},
            },
        )
        assert sent["text"]["format"] == {
            "type": "json_schema",
            "name": "route",
            "schema": schema,
            "strict": True,
        }

    @respx.mock
    async def test_json_object_passes_through(self) -> None:
        sent = await self._sent(False, response_format={"type": "json_object"})
        assert sent["text"]["format"] == {"type": "json_object"}


class TestTools:
    @respx.mock
    async def test_flat_shape_strict_and_byte_stable(self) -> None:
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=_response(CALL)))
        adapter = _adapter()
        try:
            await adapter.complete_with_tools(USER, [SEARCH])
            await adapter.complete_with_tools(USER, [SEARCH], allowed_tools=["search"])
        finally:
            await adapter.close()
        first, second = _sent(route, 0), _sent(route, 1)
        assert first["tools"] == [{"type": "function", **SEARCH["function"], "strict": True}]
        assert json.dumps(first["tools"]) == json.dumps(second["tools"])
        assert first["tool_choice"] == "auto"
        assert second["tool_choice"] == {
            "type": "allowed_tools",
            "mode": "auto",
            "tools": [{"type": "function", "name": "search"}],
        }


class TestImages:
    @respx.mock
    async def test_input_parts(self) -> None:
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=_response()))
        await _call(
            "complete",
            [
                ChatMessage(
                    role=Role.user,
                    content="describe",
                    images=(ImagePart(PNG, "image/png"), ImagePart(b"2", "image/jpeg", "high")),
                )
            ],
        )
        assert _sent(route)["input"][0]["content"] == [
            {"type": "input_text", "text": "describe"},
            {"type": "input_image", "image_url": f"data:image/png;base64,{PNG_B64}"},
            {
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{base64.b64encode(b'2').decode()}",
                "detail": "high",
            },
        ]

    async def test_stream_and_tools_reject_images(self) -> None:
        msgs = [ChatMessage(role=Role.user, content="x", images=(ImagePart(PNG, "image/png"),))]
        adapter = _adapter()
        try:
            with pytest.raises(ValueError):
                adapter.stream(msgs)
            with pytest.raises(ValueError):
                await adapter.complete_with_tools(msgs, [SEARCH])
        finally:
            await adapter.close()


def _assistant(opaque: tuple[OpaqueItem, ...] | None) -> ChatMessage:
    return ChatMessage(
        role=Role.assistant,
        tool_calls=(ToolCallRef(id="call_1", name="search", arguments='{"q":"x"}'),),
        opaque=opaque,
    )


class TestReplay:
    @staticmethod
    async def _input(msg: ChatMessage) -> list[dict]:
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=_response(CALL)))
        tool_out = ChatMessage(role=Role.tool, content="r", tool_call_id="call_1")
        await _call("complete_with_tools", [*USER, msg, tool_out], [SEARCH])
        return _sent(route)["input"][1:-1]

    @respx.mock
    async def test_same_model_replays_verbatim_in_order(self) -> None:
        opaque = (
            OpaqueItem("openai", "gpt-5-mini", REASONING),
            OpaqueItem("openai", "gpt-5-mini", CALL),
        )
        assert await self._input(_assistant(opaque)) == [REASONING, CALL]

    @respx.mock
    async def test_other_model_falls_back_to_plain_calls(self) -> None:
        opaque = (OpaqueItem("openai", "gpt-5.2", REASONING), OpaqueItem("openai", "gpt-5.2", CALL))
        plain = [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "search",
                "arguments": '{"q":"x"}',
            }
        ]
        assert await self._input(_assistant(opaque)) == plain
        assert await self._input(_assistant(None)) == plain

    @respx.mock
    async def test_opaque_without_function_call_is_ignored(self) -> None:
        sent = await self._input(_assistant((OpaqueItem("openai", "gpt-5-mini", REASONING),)))
        assert [i["type"] for i in sent] == ["function_call"]


class TestOutputParsing:
    @respx.mock
    async def test_tool_turn(self) -> None:
        pending = {**CALL, "call_id": "call_2", "id": "fc_2", "status": "in_progress"}
        respx.post(URL).mock(
            return_value=httpx.Response(200, json=_response(REASONING, CALL, pending))
        )
        turn = await _call("complete_with_tools", USER, [SEARCH])
        assert turn.tool_calls == [ToolCallRef(id="call_1", name="search", arguments='{"q":"x"}')]
        assert [o.payload["type"] for o in turn.opaque] == ["reasoning", "function_call"]
        assert turn.opaque[0].payload["encrypted_content"] == "enc-blob"
        assert all(o.model == "gpt-5-mini" for o in turn.opaque)
        assert turn.reasoning_summary == "Need the segment table."
        assert turn.finish_reason == "tool_calls"
        assert turn.meta.reasoning_context == "current_turn"
        assert turn.meta.served_model == "gpt-5-mini-2025-08-07"

    @respx.mock
    async def test_message_items_stay_out_of_opaque(self) -> None:
        respx.post(URL).mock(return_value=httpx.Response(200, json=_response(_message("hi"), CALL)))
        turn = await _call("complete_with_tools", USER, [SEARCH])
        assert turn.text == "hi"
        assert [o.payload["type"] for o in turn.opaque] == ["function_call"]

    @respx.mock
    async def test_refusal(self) -> None:
        respx.post(URL).mock(
            return_value=httpx.Response(200, json=_response(_message("can't", "refusal")))
        )
        resp = await _call("complete", USER)
        assert resp.text == "can't" and resp.finish_reason == "refusal"

    @pytest.mark.parametrize(
        ("body", "expected"),
        [
            (_response(), "stop"),
            (_response(CALL), "tool_calls"),
            (_response(REASONING, status="incomplete", incomplete="max_output_tokens"), "length"),
            (_response(status="incomplete", incomplete="content_filter"), "content_filter"),
        ],
    )
    @respx.mock
    async def test_finish_reason(self, body: dict, expected: str) -> None:
        respx.post(URL).mock(return_value=httpx.Response(200, json=body))
        assert (await _call("complete", USER)).finish_reason == expected

    @respx.mock
    async def test_failed_status_raises(self) -> None:
        body = _response(status="failed")
        body["error"] = {"code": "rate_limit_exceeded", "message": "slow down"}
        respx.post(URL).mock(return_value=httpx.Response(200, json=body))
        with pytest.raises(LLMRateLimitError):
            await _call("complete", USER)

    @respx.mock
    async def test_usage_to_stats(self) -> None:
        respx.post(URL).mock(return_value=httpx.Response(200, json=_response()))
        stats = (await _call("complete", USER)).stats
        assert (stats.input_tokens, stats.cached_input_tokens, stats.cache_write_tokens) == (
            100,
            40,
            50,
        )
        assert (stats.output_tokens, stats.reasoning_tokens, stats.total_tokens) == (20, 12, 120)
        assert stats.cost_usd is not None


def _degraded(reason: str) -> float:
    return LLM_DEGRADED_RETRIES.labels("gpt-5-mini", reason)._value.get()


class TestDegradedRetries:
    @respx.mock
    async def test_summary_unavailable_retries_once_and_sticks(self) -> None:
        before = _degraded("summary_unavailable")
        route = respx.post(URL).mock(
            side_effect=[
                _bad_request(
                    "unsupported_value", "reasoning.summary", "organization must be verified"
                ),
                httpx.Response(200, json=_response()),
                httpx.Response(200, json=_response()),
            ]
        )
        adapter = _adapter()
        try:
            await adapter.complete(USER, reasoning_effort="low", reasoning_summary="auto")
            await adapter.complete(USER, reasoning_effort="low", reasoning_summary="auto")
        finally:
            await adapter.close()
        assert "summary" in _sent(route, 0)["reasoning"]
        assert _sent(route, 1)["reasoning"] == {"effort": "low"}
        assert _sent(route, 2)["reasoning"] == {"effort": "low"}
        assert mod._summary_unavailable is True
        assert _degraded("summary_unavailable") == before + 1

    @respx.mock
    async def test_replay_rejected_retries_without_opaque(self) -> None:
        before = _degraded("reasoning_replay")
        route = respx.post(URL).mock(
            side_effect=[
                _bad_request("invalid_encrypted_content"),
                httpx.Response(200, json=_response(CALL)),
            ]
        )
        opaque = (
            OpaqueItem("openai", "gpt-5-mini", REASONING),
            OpaqueItem("openai", "gpt-5-mini", CALL),
        )
        msgs = [
            *USER,
            _assistant(opaque),
            ChatMessage(role=Role.tool, content="r", tool_call_id="call_1"),
        ]
        await _call("complete_with_tools", msgs, [SEARCH])
        assert _sent(route, 0)["input"][1] == REASONING
        assert [i.get("type") for i in _sent(route, 1)["input"][1:]] == [
            "function_call",
            "function_call_output",
        ]
        assert _degraded("reasoning_replay") == before + 1

    @respx.mock
    async def test_second_failure_propagates(self) -> None:
        respx.post(URL).mock(return_value=_bad_request("invalid_encrypted_content"))
        with pytest.raises(LLMReasoningReplayError):
            await _call("complete_with_tools", USER, [SEARCH])


def _sse(*events: dict) -> bytes:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def _delta(text: str, kind: str = "output_text") -> dict:
    return {
        "type": f"response.{kind}.delta",
        "item_id": "msg_1",
        "output_index": 0,
        "content_index": 0,
        "delta": text,
        "sequence_number": 0,
    }


def _terminal(body: dict, kind: str = "completed") -> dict:
    return {"type": f"response.{kind}", "response": body, "sequence_number": 9}


ERROR = {"type": "error", "code": "server_error", "message": "boom", "sequence_number": 1}


async def _stream(body: bytes, adapter: OpenAIResponsesAdapter | None = None) -> list:
    respx.post(URL).mock(
        return_value=httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=body
        )
    )
    adapter = adapter or _adapter()
    try:
        return [c async for c in adapter.stream(USER)]
    finally:
        await adapter.close()


class TestStreaming:
    @respx.mock
    async def test_deltas_then_one_final_chunk(self) -> None:
        summary = {
            "type": "response.reasoning_summary_text.delta",
            "item_id": "rs_1",
            "output_index": 0,
            "summary_index": 0,
            "delta": "thinking",
            "sequence_number": 1,
        }
        chunks = await _stream(
            _sse(
                summary,
                _delta("he"),
                _delta("llo"),
                _terminal(_response(REASONING, _message("hello"))),
            )
        )
        assert [c.text for c in chunks] == ["he", "llo", ""]
        assert sum(c.is_final for c in chunks) == 1 and chunks[-1].is_final
        final = chunks[-1]
        assert final.finish_reason == "stop"
        assert final.reasoning_summary == "Need the segment table."
        assert final.stats.cache_write_tokens == 50 and final.stats.ttft_ms is not None

    @respx.mock
    async def test_incomplete_is_length(self) -> None:
        body = _response(_message("par"), status="incomplete", incomplete="max_output_tokens")
        chunks = await _stream(_sse(_delta("par"), _terminal(body, "incomplete")))
        assert chunks[-1].is_final and chunks[-1].finish_reason == "length"

    @respx.mock
    async def test_refusal_delta_is_text(self) -> None:
        body = _response(_message("no", "refusal"))
        chunks = await _stream(_sse(_delta("no", "refusal"), _terminal(body)))
        assert chunks[0].text == "no" and chunks[-1].finish_reason == "refusal"

    @respx.mock
    async def test_dropped_stream_raises(self) -> None:
        with pytest.raises(LLMConnectionError):
            await _stream(_sse(_delta("partial")))

    @pytest.mark.parametrize(
        ("code", "exc"),
        [
            ("rate_limit_exceeded", LLMRateLimitError),
            ("server_error", LLMServerError),
            ("invalid_prompt", LLMInvalidRequestError),
            ("something_new", LLMServerError),
        ],
    )
    @respx.mock
    async def test_error_event_mapping(self, code: str, exc: type) -> None:
        with pytest.raises(exc):
            await _stream(_sse({**ERROR, "code": code}))

    @respx.mock
    async def test_failed_event_raises(self) -> None:
        body = _response(status="failed")
        body["error"] = {"code": "server_error", "message": "boom"}
        with pytest.raises(LLMServerError):
            await _stream(_sse(_terminal(body, "failed")))

    @respx.mock
    async def test_error_after_text_propagates(self) -> None:
        with pytest.raises(LLMServerError):
            await _stream(_sse(_delta("partial"), ERROR))
