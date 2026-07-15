from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class EvalQuestion(BaseModel):
    qid: str
    question: str
    kind: str
    answers: list[str]
    reference_pools: list[list[str]]


class ExcludedEntry(BaseModel):
    qid: str
    reason: str


class RetrievalMetrics(BaseModel):
    model_config = {"extra": "allow"}


class CorrectnessResult(BaseModel):
    correct: bool
    reason: str


class PerQuestionResult(BaseModel):
    qid: str
    question: str
    kind: str
    expected_answers: list[str]
    reference_pools: list[list[str]]
    route: str | None = None
    excluded_reason: str | None = None
    retrieved_page_keys: list[str] = []
    metrics: dict[str, float] = {}
    answer: str | None = None
    citation_spans: list[dict] = []
    correctness: CorrectnessResult | None = None
    judge: dict[str, Any] | None = None
    latency_s: float | None = None
    usage: dict[str, Any] | None = None
    # Agentic-path-only fields (populated by run_agent.py; Stage 0.5 baseline signal)
    query_shape: str | None = None
    agent_meta: dict[str, Any] | None = None
    observations_count: int | None = None
    confidence_counts: dict[str, int] | None = None
    gaps_count: int | None = None
    ref_id_to_chunk_id: dict[str, str] = {}


class RunManifest(BaseModel):
    timestamp: str
    git_sha: str
    test_set: str
    test_set_hash: str
    model: str
    reasoning_effort: str | None = None
    verbosity: str | None = None
    max_tokens: int | None = None
    judge_model: str
    k_values: list[int]
    run_description: str | None = None
    total_questions: int
    evaluated: int
    excluded: list[ExcludedEntry]


class AggregateMetrics(BaseModel):
    retrieval: dict[str, float] = {}
    correctness: dict[str, Any] = {}
    judge: dict[str, float] = {}
    hallucination: dict[str, Any] = {}
    # Agentic-path-only, cut by query_shape (Stage 0 — docs/agentic_pattern_evolution_upd.md)
    agent: dict[str, Any] = {}


class RunOutput(BaseModel):
    manifest: RunManifest
    aggregate: AggregateMetrics
    per_question: list[PerQuestionResult]
