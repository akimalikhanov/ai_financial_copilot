"""Central Prometheus metric definitions.

Every metric lives here so labels stay consistent and cardinality stays bounded.
Labels must be low-cardinality: `endpoint` is a route template (never a raw path),
`model` is a fixed enum from models.yaml, and tool/decision/type are bounded sets.
Never put request_id / user_id / raw paths in a label — those belong in logs.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# --- HTTP ---
HTTP_REQUESTS = Counter(
    "http_requests_total",
    "HTTP requests",
    ["method", "endpoint", "status"],  # endpoint = ROUTE TEMPLATE, never raw path
)
HTTP_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP latency",
    ["method", "endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)
HTTP_IN_PROGRESS = Gauge(
    "http_requests_in_progress",
    "In-flight requests",
    ["method", "endpoint"],
)

# --- SSE streams ---
# Naive middleware records duration/in-progress at call_next return, which for a
# StreamingResponse is when headers are ready, not when the body finishes — so these are
# instrumented directly in the generator instead of via HTTP_DURATION/HTTP_IN_PROGRESS.
SSE_STREAMS_OPEN = Gauge("sse_streams_open", "Currently open SSE streams", ["endpoint"])
SSE_STREAM_DURATION = Histogram(
    "sse_stream_duration_seconds",
    "SSE stream lifetime",
    ["endpoint", "outcome"],
    # Answers run 12s mean / 43s p95, so the default 10s-capped buckets are useless here.
    buckets=(1, 5, 10, 30, 60, 120, 300, 600),
)

# /readyz needs this count to decide whether the pod is at capacity, and the Gauge cannot
# supply it: the API runs with PROMETHEUS_MULTIPROC_DIR set (Dockerfile.api), where a Gauge
# holds no readable in-process value — reading it means touching prometheus_client internals.
# A plain int in this process is the honest source for a local capacity check anyway; the
# Gauge stays the source for scraping. Single-threaded event loop, so no lock is needed.
_open_streams: dict[str, int] = {}


def sse_stream_opened(endpoint: str) -> None:
    """Record an SSE stream opening. Bumps the scrape Gauge and the /readyz counter together."""
    SSE_STREAMS_OPEN.labels(endpoint).inc()
    _open_streams[endpoint] = _open_streams.get(endpoint, 0) + 1


def sse_stream_closed(endpoint: str) -> None:
    """Record an SSE stream closing. Must be called from a `finally` so the count cannot leak."""
    SSE_STREAMS_OPEN.labels(endpoint).dec()
    _open_streams[endpoint] = max(0, _open_streams.get(endpoint, 0) - 1)


def open_stream_count() -> int:
    """Total SSE streams open in this process, across endpoints."""
    return sum(_open_streams.values())


CHAT_QUEUE_WAIT = Histogram(
    "chat_queue_wait_seconds",
    "Enqueue -> task start",
    buckets=(0.1, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300),
)

# --- Celery ---
CELERY_TASKS = Counter("celery_tasks_total", "Celery tasks", ["task_name", "state"])
CELERY_DURATION = Histogram(
    "celery_task_duration_seconds",
    "Task duration",
    ["task_name"],
    # ingest_document runs minute-scale; default buckets cap at 10s and dump
    # every ingestion into +Inf, breaking histogram_quantile (NaN).
    buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600),
)
CELERY_QUEUE = Gauge("celery_queue_length", "Queue depth", ["queue_name"])

# --- Agentic RAG ---
RAG_RETRIEVAL = Histogram(
    "rag_retrieval_duration_seconds",
    "Retrieval latency",
    ["stage"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
)
RAG_CHUNKS = Histogram("rag_chunks_retrieved", "Chunks per query", ["retriever"])
RAG_CHUNKS_UNHYDRATED = Counter(
    "rag_chunks_unhydrated_total",
    "Retrieved chunks skipped at assembly: no Postgres row (stale Qdrant/OpenSearch entry)",
)
RAG_CONTEXT_TOKENS = Histogram(
    "rag_context_tokens",
    "Context tokens",
    buckets=(256, 512, 1024, 2048, 4096, 8192, 16384),
)
RAG_CITATIONS = Histogram(
    "rag_citations_per_response",
    "Citations per response",
    buckets=(0, 1, 2, 3, 5, 8, 13),
)
AGENT_ITERATIONS = Histogram(
    "agent_loop_iterations",
    "Agent loop steps",
    buckets=(1, 2, 3, 4, 5, 8),
)
AGENT_TOOL_CALLS = Counter("agent_tool_calls_total", "Tool calls", ["tool", "status"])
CITATION_REFS_DROPPED = Counter(
    "citation_refs_dropped_total",
    "Finding citations dropped (no excerpt in synthesis context)",
)
REVIVED_CHUNKS_PER_TURN = Histogram(
    "agent_revived_chunks_per_turn",
    "Evicted-then-re-returned chunks re-emitted under their original label",
    buckets=(0, 1, 2, 3, 5, 10),
)
AGENT_TOOL_DURATION = Histogram(
    "agent_tool_duration_seconds",
    "Tool latency",
    ["tool"],
    # search_documents wraps full RAG retrieve+rerank and can exceed 10s.
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)
ROUTER_DECISIONS = Counter("query_router_decisions_total", "Router decisions", ["decision"])
# outcome: carried | none | dropped_hop_cap | dropped_scope
FOLLOWUP_FINDINGS_CARRIED = Counter(
    "followup_findings_carried_total",
    "Prior-turn findings block reuse on non-retrieval turns",
    ["outcome"],
)
# grounded: whether the direct answer had a carried block behind it
FOLLOWUP_DIRECT_ANSWER = Counter(
    "router_followup_direct_answer_total",
    "Turns answered without retrieval",
    ["grounded"],
)
GUARDRAIL_BLOCKS = Counter("guardrail_blocks_total", "Guardrail blocks", ["type"])
PIPELINE_ERRORS = Counter("chat_pipeline_errors_total", "Chat pipeline failures", ["stage"])

# --- LLM cost/tokens ---
LLM_TOKENS = Counter("llm_tokens_total", "Tokens", ["direction", "model"])
LLM_COST = Counter("llm_cost_usd_total", "Cost USD", ["model"])
LLM_CACHE_HIT_TOKENS = Counter("llm_cache_hit_tokens_total", "Cached input tokens", ["model"])

# --- Ingestion ---
INGESTION_DOCUMENTS = Counter("ingestion_documents_total", "Documents processed", ["status"])
INGESTION_CHUNKS = Histogram(
    "ingestion_chunks_per_document",
    "Chunks per document",
    buckets=(10, 25, 50, 100, 200, 500, 1000),
)
PICTURE_ENRICHER = Counter(
    "picture_enricher_pictures_total",
    "Pictures by lane and outcome",
    # status: described | empty | skipped | failed. "empty" is the signature of a starved
    # completion budget (reasoning spends it before any JSON is emitted) — graph it.
    ["lane", "status"],
)
PICTURE_ENRICHER_DURATION = Histogram(
    "picture_enricher_batch_seconds",
    "Per-batch latency",
    ["lane"],
    buckets=(0.5, 1, 2.5, 5, 10, 30, 60),
)
INGESTION_DURATION = Histogram(
    "ingestion_stage_duration_seconds",
    "Ingestion stage latency",
    ["stage"],  # parse | chunk | embed | upsert_qdrant | upsert_opensearch
    # Stages span ms-scale upserts to minute-scale Docling parses; default
    # buckets top out at 10s and dump every parse into +Inf (breaks quantiles).
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300),
)
