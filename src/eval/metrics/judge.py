from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError

from src.schemas.retrieval import AnswerCitationSpan, RAGContext
from src.services.llm_adapters.base_adapter import ChatMessage, Role
from src.services.llm_router import get_router
from src.utils.json_schema import build_response_format

logger = logging.getLogger(__name__)

_JUDGE_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "judge_system.yaml"


class _ScoreField(BaseModel):
    score: int = Field(ge=1, le=5)
    justification: str


class _UnsupportedClaim(BaseModel):
    claim: str
    cited_refs: list[str] = []
    reason: str


class JudgeOutput(BaseModel):
    faithfulness: _ScoreField
    relevance: _ScoreField
    citation_accuracy: _ScoreField
    completeness: _ScoreField
    unsupported_claims: list[_UnsupportedClaim] = []


def _load_judge_system_prompt() -> str:
    with open(_JUDGE_PROMPT_PATH) as f:
        data = yaml.safe_load(f)
    return data["template"]


def _format_retrieved_context(rag_context: RAGContext) -> str:
    if not rag_context.items:
        return "(no context retrieved)"
    lines = []
    for item in rag_context.items:
        pages = ",".join(str(p) for p in item.citation.page_numbers)
        snippet = item.citation.snippet or item.prompt_text
        lines.append(f"[{item.ref_id}] {item.citation.document_name} p{pages}: {snippet}")
    return "\n\n".join(lines)


def _build_user_message(
    question: str,
    rag_context: RAGContext,
    answer: str,
    gold_answer: str,
) -> str:
    return (
        f"QUESTION:\n{question}\n\n"
        f"RETRIEVED_CONTEXT:\n{_format_retrieved_context(rag_context)}\n\n"
        f"MODEL_ANSWER:\n{answer}\n\n"
        f"GOLD_ANSWER:\n{gold_answer}"
    )


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _parse_judge_response(raw: str) -> JudgeOutput | None:
    text = raw.strip()
    m = _JSON_FENCE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        data = json.loads(text)
        return JudgeOutput.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as e:
        logger.warning("judge_parse_failed: %s raw=%r", e, raw[:300])
        return None


def hallucination_rate(
    judge: JudgeOutput,
    citation_spans: list[AnswerCitationSpan] | tuple[AnswerCitationSpan, ...],
) -> float:
    return len(judge.unsupported_claims) / max(1, len(citation_spans))


async def judge_one(
    *,
    question: str,
    rag_context: RAGContext,
    answer: str,
    gold_answer: str,
    model_id: str,
    system_prompt: str | None = None,
) -> JudgeOutput | None:
    """Single-call judge for one question. Returns None on parse/LLM failure."""
    system = system_prompt or _load_judge_system_prompt()
    user = _build_user_message(question, rag_context, answer, gold_answer)
    messages = [
        ChatMessage(role=Role.system, content=system),
        ChatMessage(role=Role.user, content=user),
    ]
    response_format = build_response_format("judge_output", JudgeOutput.model_json_schema())
    llm = get_router().get(model_id)
    try:
        resp = await llm.complete(
            messages=messages,
            temperature=0.0,
            response_format=response_format,
        )
    except Exception as e:
        logger.exception("judge_llm_error: %s", e)
        return None
    return _parse_judge_response(resp.text or "")


async def judge_many(
    items: list[dict[str, Any]],
    *,
    model_id: str,
    concurrency: int = 8,
) -> list[JudgeOutput | None]:
    """Judge many questions concurrently.

    Each item must contain: question, rag_context, answer, gold_answer.
    Preserves input order. Results may include None for failed calls.
    """
    system_prompt = _load_judge_system_prompt()
    sem = asyncio.Semaphore(concurrency)

    async def _run(item: dict[str, Any]) -> JudgeOutput | None:
        async with sem:
            return await judge_one(
                question=item["question"],
                rag_context=item["rag_context"],
                answer=item["answer"],
                gold_answer=item["gold_answer"],
                model_id=model_id,
                system_prompt=system_prompt,
            )

    return await asyncio.gather(*(_run(item) for item in items))
