from __future__ import annotations

import base64
import logging
from collections.abc import AsyncGenerator, Mapping, Sequence
from dataclasses import replace
from typing import Any

from src.observability.metrics import LLM_DEGRADED_RETRIES
from src.services.llm_runtime.exception_mapper import map_openai_error, map_openai_response_error
from src.services.llm_runtime.exceptions import (
    LLMConnectionError,
    LLMError,
    LLMFeatureUnavailableError,
    LLMReasoningReplayError,
)
from src.utils.config import get_llm_prompt_cache_key_enabled
from src.utils.llm_utils import (
    calc_cost_openai,
    compute_tps,
    elapsed_ms,
    get_pricing_for_model,
    now_ms,
)

from .base_adapter import (
    AssistantTurnResult,
    ChatMessage,
    ChatRequest,
    ImagePart,
    LLMAdapter,
    LLMResponse,
    LLMResponseStats,
    LLMStreamChunk,
    OpaqueItem,
    ResponseMeta,
    Role,
    ToolCallRef,
    reject_images,
)
from .openai_adapter import make_openai_client

logger = logging.getLogger(__name__)

# Set on the first "summaries need a verified org" 400; this process stops asking.
_summary_unavailable = False


def _replayable(m: ChatMessage, model: str) -> list[dict[str, Any]]:
    """The message's own opaque items for this exact model, or nothing. Items without a
    function call can't stand in for `tool_calls`, so they're ignored."""
    items = [o.payload for o in (m.opaque or ()) if o.provider == "openai" and o.model == model]
    return items if any(i.get("type") == "function_call" for i in items) else []


def _image_part(img: ImagePart) -> dict[str, Any]:
    b64 = base64.b64encode(img.data).decode("ascii")
    part: dict[str, Any] = {
        "type": "input_image",
        "image_url": f"data:{img.mime_type};base64,{b64}",
    }
    if img.detail != "auto":
        part["detail"] = img.detail
    return part


def _content(m: ChatMessage) -> str | list[dict[str, Any]]:
    if not m.images:
        return m.content or ""
    parts: list[dict[str, Any]] = [{"type": "input_text", "text": m.content}] if m.content else []
    parts.extend(_image_part(img) for img in m.images)
    return parts


def _text_format(fmt: Mapping[str, Any]) -> dict[str, Any]:
    """Chat Completions `response_format` → Responses `text.format` (json_schema is flat)."""
    if fmt.get("type") == "json_schema":
        return {"type": "json_schema", **fmt["json_schema"]}
    return dict(fmt)


def _convert_tools(tools: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    # strict is explicit: Responses defaults to strict but silently falls back to non-strict.
    return [{"type": "function", **t["function"], "strict": True} for t in tools]


class OpenAIResponsesAdapter(LLMAdapter):
    """OpenAI over /v1/responses, stateless (store=false).

    The loop's transcript is the source of truth. Reasoning items come back encrypted
    and ride on ChatMessage.opaque; this adapter is the only code that reads them."""

    provider_name: str = "openai"
    api: str = "responses"

    def __init__(
        self,
        *,
        default_model: str,
        capabilities: Mapping[str, Any] | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        super().__init__(default_model=default_model)
        self._reasoning = bool((capabilities or {}).get("reasoning", False))
        self._client = make_openai_client(api_key=api_key, base_url=base_url)

    async def close(self) -> None:
        await self._client.close()

    # ---- request ----

    def _serialize_input(
        self, messages: Sequence[ChatMessage], model: str, *, replay: bool
    ) -> tuple[str | None, list[dict[str, Any]]]:
        instructions: str | None = None
        items: list[dict[str, Any]] = []
        for i, m in enumerate(messages):
            if i == 0 and m.role == Role.system:
                # Keeps today's prefix order: instructions, then tools, then items.
                instructions = m.content or ""
            elif m.role in (Role.system, Role.developer):
                items.append({"role": "developer", "content": _content(m)})
            elif m.role == Role.user:
                items.append({"role": "user", "content": _content(m)})
            elif m.role == Role.assistant and m.tool_calls:
                items.extend(
                    (replay and _replayable(m, model))
                    or [
                        {
                            "type": "function_call",
                            "call_id": tc.id,
                            "name": tc.name,
                            "arguments": tc.arguments,
                        }
                        for tc in m.tool_calls
                    ]
                )
            elif m.role == Role.assistant:
                items.append({"role": "assistant", "content": m.content or ""})
            elif m.role == Role.tool:
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": m.tool_call_id,
                        "output": m.content or "",
                    }
                )
        return instructions, items

    def _build_kwargs(self, req: ChatRequest, *, replay: bool) -> dict[str, Any]:
        extra = dict(req.extra_params)
        instructions, items = self._serialize_input(req.messages, req.model, replay=replay)
        # Invariants, not tunables: the transcript is the only conversation state
        # (store=False), and silently dropping old items would hide S-labels the model can
        # cite (truncation="disabled"). Never set previous_response_id, conversation or
        # context_management: each moves state off the transcript or rewrites the prefix.
        kw: dict[str, Any] = {
            "model": req.model,
            "input": items,
            "store": False,
            "truncation": "disabled",
        }
        if instructions is not None:
            kw["instructions"] = instructions
        if req.max_tokens is not None:
            kw["max_output_tokens"] = req.max_tokens
        if req.temperature is not None and not self._reasoning:
            kw["temperature"] = req.temperature

        effort = extra.pop("reasoning_effort", None)
        summary = extra.pop("reasoning_summary", None)
        context = extra.pop("reasoning_context", None)
        verbosity = extra.pop("verbosity", None)
        fmt = extra.pop("response_format", None)
        extra.pop("enable_thinking", None)  # Qwen-only
        if not get_llm_prompt_cache_key_enabled():
            extra.pop("prompt_cache_key", None)

        reasoning: dict[str, Any] = {}
        text: dict[str, Any] = {}
        # Empty strings (from `${X:-}` in models.yaml) mean unset.
        if self._reasoning:
            if effort:
                reasoning["effort"] = effort
            if summary and not _summary_unavailable:
                reasoning["summary"] = summary
            if context:
                reasoning["context"] = context
            if verbosity:
                text["verbosity"] = verbosity
        if fmt:
            text["format"] = _text_format(fmt)
        if reasoning:
            kw["reasoning"] = reasoning
        if text:
            kw["text"] = text
        kw.update(extra)
        return kw

    async def _create(
        self, req: ChatRequest, extra: Mapping[str, Any] | None = None, *, stream: bool = False
    ) -> Any:
        """One call, resent at most once per degraded feature: without the summary when the
        account can't have one, without opaque items when a replayed one is rejected."""
        global _summary_unavailable
        replay = True
        while True:
            kw = self._build_kwargs(req, replay=replay)
            kw.update(extra or {})
            if stream:
                kw["stream"] = True
            try:
                return await self._client.responses.create(**kw)
            except Exception as e:
                err = map_openai_error(e, provider=self.provider_name, model=req.model)
                if isinstance(err, LLMFeatureUnavailableError) and "summary" in kw.get(
                    "reasoning", {}
                ):
                    if not _summary_unavailable:
                        logger.warning("llm_reasoning_summary_unavailable model=%s", req.model)
                    _summary_unavailable = True
                    LLM_DEGRADED_RETRIES.labels(req.model, "summary_unavailable").inc()
                    continue
                if isinstance(err, LLMReasoningReplayError) and replay:
                    logger.warning("llm_reasoning_replay_rejected model=%s", req.model)
                    replay = False
                    LLM_DEGRADED_RETRIES.labels(req.model, "reasoning_replay").inc()
                    continue
                raise err from e

    # ---- response ----

    def _stats(
        self, usage: Any, *, model: str, latency_ms: float, ttft_ms: float | None = None
    ) -> LLMResponseStats:
        in_details = getattr(usage, "input_tokens_details", None)
        out_details = getattr(usage, "output_tokens_details", None)
        output_tokens = getattr(usage, "output_tokens", None)
        stats = LLMResponseStats(
            input_tokens=getattr(usage, "input_tokens", None),
            cached_input_tokens=getattr(in_details, "cached_tokens", None),
            cache_write_tokens=getattr(in_details, "cache_write_tokens", None),
            output_tokens=output_tokens,
            reasoning_tokens=getattr(out_details, "reasoning_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            tps=compute_tps(output_tokens=output_tokens, latency_ms=latency_ms, ttft_ms=ttft_ms),
        )
        pricing = get_pricing_for_model("openai", model)
        if pricing is not None:
            cost = calc_cost_openai(stats, pricing)
            if cost is not None:
                stats = replace(stats, cost_usd=cost)
        return stats

    def _parse(self, resp: Any, model: str) -> dict[str, Any]:
        """Walk `resp.output` in order. Opaque keeps reasoning and completed function-call
        items only: model prose never enters the transcript."""
        if resp.status == "failed":
            error = resp.error
            raise map_openai_response_error(
                getattr(error, "code", None),
                getattr(error, "message", None) or "response failed",
                provider=self.provider_name,
                model=model,
            )
        text: list[str] = []
        summary: list[str] = []
        calls: list[ToolCallRef] = []
        opaque: list[OpaqueItem] = []
        refused = False
        for item in resp.output:
            if item.type == "reasoning":
                summary.extend(s.text for s in item.summary or ())
                opaque.append(OpaqueItem("openai", model, item.model_dump(exclude_none=True)))
            elif item.type == "function_call" and item.status in (None, "completed"):
                # call_id, not the fc_ item id: it's what function_call_output echoes back.
                calls.append(ToolCallRef(id=item.call_id, name=item.name, arguments=item.arguments))
                opaque.append(OpaqueItem("openai", model, item.model_dump(exclude_none=True)))
            elif item.type == "message":
                for part in item.content:
                    if part.type == "output_text":
                        text.append(part.text)
                    elif part.type == "refusal":
                        text.append(part.refusal)
                        refused = True

        incomplete = resp.incomplete_details.reason if resp.incomplete_details else None
        if resp.status == "incomplete":
            finish_reason = "length" if incomplete == "max_output_tokens" else incomplete
        elif refused:
            finish_reason = "refusal"
        else:
            finish_reason = "tool_calls" if calls else "stop"
        return {
            "text": "".join(text),
            "tool_calls": calls,
            "opaque": tuple(opaque) or None,
            "finish_reason": finish_reason,
            "reasoning_summary": "\n\n".join(summary) or None,
            "meta": ResponseMeta(
                response_id=resp.id,
                served_model=resp.model,
                reasoning_context=resp.reasoning.context if resp.reasoning else None,
                service_tier=resp.service_tier,
                status=resp.status,
                incomplete_reason=incomplete,
            ),
        }

    # ---- methods ----

    async def _complete(self, req: ChatRequest) -> LLMResponse:
        start_ms = now_ms()
        resp = await self._create(req)
        latency_ms = elapsed_ms(start_ms)
        out = self._parse(resp, req.model)
        return LLMResponse(
            text=out["text"].strip(),
            raw=resp,
            stats=self._stats(resp.usage, model=req.model, latency_ms=latency_ms),
            finish_reason=out["finish_reason"],
            reasoning_summary=out["reasoning_summary"],
            meta=out["meta"],
        )

    async def complete_with_tools(
        self,
        messages: Sequence[ChatMessage],
        tools: list[dict[str, Any]],
        allowed_tools: list[str] | None = None,
        **kwargs: Any,
    ) -> AssistantTurnResult:
        reject_images(messages, f"{self.__class__.__name__}.complete_with_tools")
        req = ChatRequest(
            messages=tuple(messages),
            model=kwargs.pop("model", None) or self.default_model,
            temperature=kwargs.pop("temperature", None),
            max_tokens=kwargs.pop("max_tokens", None),
            extra_params=kwargs,
        )
        extra = {
            "tools": _convert_tools(tools),
            # `allowed_tools` restricts the callable subset while `tools` stays
            # byte-identical, so the cached prompt prefix survives the restriction.
            "tool_choice": (
                {
                    "type": "allowed_tools",
                    "mode": "auto",
                    "tools": [{"type": "function", "name": n} for n in allowed_tools],
                }
                if allowed_tools
                else "auto"
            ),
            "parallel_tool_calls": True,
            # Stateless carryover is impossible without it; "off" is reasoning_context.
            "include": ["reasoning.encrypted_content"],
        }
        start_ms = now_ms()
        resp = await self._create(req, extra)
        latency_ms = elapsed_ms(start_ms)
        out = self._parse(resp, req.model)
        return AssistantTurnResult(
            text=out["text"],
            tool_calls=out["tool_calls"],
            stats=self._stats(resp.usage, model=req.model, latency_ms=latency_ms),
            finish_reason=out["finish_reason"],
            opaque=out["opaque"],
            reasoning_summary=out["reasoning_summary"],
            meta=out["meta"],
        )

    async def _stream(self, req: ChatRequest) -> AsyncGenerator[LLMStreamChunk, None]:
        start_ms = now_ms()
        first_token_ms: float | None = None
        stream = await self._create(req, stream=True)
        try:
            async for ev in stream:
                # Refusal text is yielded too, so the user sees it instead of an empty answer.
                if ev.type in ("response.output_text.delta", "response.refusal.delta"):
                    if first_token_ms is None:
                        first_token_ms = now_ms()
                    yield LLMStreamChunk(text=ev.delta, raw=ev)
                elif ev.type in ("response.completed", "response.incomplete", "response.failed"):
                    resp = ev.response
                    out = self._parse(resp, req.model)  # raises on failed
                    ttft_ms = elapsed_ms(start_ms, first_token_ms) if first_token_ms else None
                    yield LLMStreamChunk(
                        text="",
                        raw=ev,
                        is_final=True,
                        stats=self._stats(
                            resp.usage,
                            model=req.model,
                            latency_ms=elapsed_ms(start_ms),
                            ttft_ms=ttft_ms,
                        ),
                        finish_reason=out["finish_reason"],
                        reasoning_summary=out["reasoning_summary"],
                        meta=out["meta"],
                    )
                    return
                elif ev.type == "error":
                    raise map_openai_response_error(
                        ev.code, ev.message, provider=self.provider_name, model=req.model
                    )
        except LLMError:
            raise
        except Exception as e:
            raise map_openai_error(e, provider=self.provider_name, model=req.model) from e
        # No terminal event: the connection dropped mid-answer.
        raise LLMConnectionError(
            "stream ended without a terminal event",
            provider=self.provider_name,
            model=req.model,
        )
