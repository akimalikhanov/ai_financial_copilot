"""Unit tests for the OpenAI adapter's wire payload (Phase 10: multimodal support).

Asserts on the JSON that actually leaves the process rather than on _serialize_msg, so a
change to how the request is assembled cannot pass these while breaking the provider call.
The same class serves vLLM (provider_name="vllm"), so this covers that path too.
"""

from __future__ import annotations

import base64
import json

import httpx
import pytest
import respx

from src.services.llm_adapters.base_adapter import ChatMessage, ImagePart, Role, ToolCallRef
from src.services.llm_adapters.openai_adapter import OpenAIAdapter

BASE_URL = "http://openai-test/v1"
PNG = b"\x89PNG\r\n\x1a\n" + b"payload"
PNG_B64 = base64.b64encode(PNG).decode("ascii")


def _adapter() -> OpenAIAdapter:
    return OpenAIAdapter(default_model="gpt-4o-mini", api_key="test-key", base_url=BASE_URL)


def _completion(content: str = "ok") -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }


def _route():
    return respx.post(f"{BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json=_completion())
    )


async def _sent_messages(messages: list[ChatMessage]) -> list[dict]:
    """Round-trip one complete() call and return the serialized messages it sent."""
    route = _route()
    adapter = _adapter()
    try:
        await adapter.complete(messages)
    finally:
        await adapter.close()
    return json.loads(route.calls.last.request.content)["messages"]


class TestTextOnlyUnchanged:
    """Phase 10 is additive: messages without images must serialize as they always have."""

    @respx.mock
    async def test_plain_text_message(self) -> None:
        sent = await _sent_messages(
            [
                ChatMessage(role=Role.system, content="sys"),
                ChatMessage(role=Role.user, content="hi"),
            ]
        )
        assert sent == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]

    @respx.mock
    async def test_assistant_tool_call_turn_keeps_empty_string_content(self) -> None:
        """content=None still becomes "", not a parts list and not a dropped key."""
        sent = await _sent_messages(
            [
                ChatMessage(
                    role=Role.assistant,
                    content=None,
                    tool_calls=(ToolCallRef(id="c1", name="search", arguments='{"q":"x"}'),),
                ),
                ChatMessage(role=Role.tool, content="result", tool_call_id="c1"),
            ]
        )
        assert sent[0]["content"] == ""
        assert sent[0]["tool_calls"][0]["function"]["name"] == "search"
        assert sent[1] == {"role": "tool", "content": "result", "tool_call_id": "c1"}


class TestImageParts:
    @respx.mock
    async def test_text_and_image_becomes_parts_list(self) -> None:
        sent = await _sent_messages(
            [
                ChatMessage(
                    role=Role.user,
                    content="describe",
                    images=(ImagePart(data=PNG, mime_type="image/png"),),
                )
            ]
        )
        assert sent[0]["content"] == [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG_B64}"}},
        ]

    @respx.mock
    async def test_image_only_message_omits_the_text_part(self) -> None:
        """An image-only turn is valid; an empty text part is noise the model has to read."""
        sent = await _sent_messages(
            [ChatMessage(role=Role.user, content=None, images=(ImagePart(PNG, "image/png"),))]
        )
        assert sent[0]["content"] == [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG_B64}"}}
        ]

    @respx.mock
    async def test_multiple_images_keep_their_order(self) -> None:
        sent = await _sent_messages(
            [
                ChatMessage(
                    role=Role.user,
                    content="two",
                    images=(
                        ImagePart(b"first", "image/png"),
                        ImagePart(b"second", "image/jpeg"),
                    ),
                )
            ]
        )
        urls = [p["image_url"]["url"] for p in sent[0]["content"] if p["type"] == "image_url"]
        assert urls == [
            f"data:image/png;base64,{base64.b64encode(b'first').decode()}",
            f"data:image/jpeg;base64,{base64.b64encode(b'second').decode()}",
        ]

    @respx.mock
    async def test_mime_type_is_carried_not_guessed(self) -> None:
        sent = await _sent_messages(
            [ChatMessage(role=Role.user, content="x", images=(ImagePart(PNG, "image/webp"),))]
        )
        assert sent[0]["content"][1]["image_url"]["url"].startswith("data:image/webp;base64,")


class TestDetail:
    """detail is the cost lever, so its presence and absence are both load-bearing."""

    @respx.mock
    @pytest.mark.parametrize("detail", ["low", "high"])
    async def test_explicit_detail_is_sent(self, detail: str) -> None:
        sent = await _sent_messages(
            [
                ChatMessage(
                    role=Role.user,
                    content="x",
                    images=(ImagePart(PNG, "image/png", detail=detail),),  # type: ignore[arg-type]
                )
            ]
        )
        assert sent[0]["content"][1]["image_url"]["detail"] == detail

    @respx.mock
    async def test_auto_omits_the_field(self) -> None:
        """ "auto" is OpenAI's own default; sending it would be noise, not a choice."""
        sent = await _sent_messages(
            [ChatMessage(role=Role.user, content="x", images=(ImagePart(PNG, "image/png"),))]
        )
        assert "detail" not in sent[0]["content"][1]["image_url"]

    @respx.mock
    async def test_parts_carry_independent_detail(self) -> None:
        sent = await _sent_messages(
            [
                ChatMessage(
                    role=Role.user,
                    content="x",
                    images=(
                        ImagePart(PNG, "image/png", "high"),
                        ImagePart(PNG, "image/png", "low"),
                    ),
                )
            ]
        )
        details = [
            p["image_url"].get("detail") for p in sent[0]["content"] if p["type"] == "image_url"
        ]
        assert details == ["high", "low"]


class TestUnsupportedSurfaces:
    """Images ride on complete() only. Silently dropping them would mean paying for a
    description of nothing; these two must fail loudly instead."""

    IMG_MSG = [ChatMessage(role=Role.user, content="x", images=(ImagePart(PNG, "image/png"),))]

    async def test_stream_rejects_images(self) -> None:
        adapter = _adapter()
        try:
            with pytest.raises(ValueError, match="does not support image content"):
                adapter.stream(self.IMG_MSG)
        finally:
            await adapter.close()

    async def test_complete_with_tools_rejects_images(self) -> None:
        adapter = _adapter()
        try:
            with pytest.raises(ValueError, match="does not support image content"):
                await adapter.complete_with_tools(self.IMG_MSG, tools=[])
        finally:
            await adapter.close()

    @respx.mock
    async def test_text_only_still_streams(self) -> None:
        """The guard must not block the text path it wraps."""
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=b'data: {"id":"1","object":"chat.completion.chunk","created":0,'
                b'"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"hi"},'
                b'"finish_reason":null}]}\n\ndata: [DONE]\n\n',
            )
        )
        adapter = _adapter()
        try:
            chunks = [c async for c in adapter.stream([ChatMessage(role=Role.user, content="x")])]
        finally:
            await adapter.close()
        assert "".join(c.text for c in chunks) == "hi"
