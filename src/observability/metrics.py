"""Central Prometheus metric definitions.

Every metric lives here so labels stay consistent and cardinality stays bounded.
Labels must be low-cardinality: `endpoint` is a route template (never a raw path),
`model` is a fixed enum from models.yaml, and tool/decision/type are bounded sets.
Never put request_id / user_id / raw paths in a label — those belong in logs.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

from src.services.llm_adapters.base_adapter import LLMResponseStats

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
    # livesum, not the default "all": under multiprocess the default keeps a per-PID
    # series (unbounded cardinality across restarts, and every query needs its own
    # sum()), and mark_process_dead only deletes live* shards — so a SIGKILLed
    # process's leaked +1 would have no cleanup path at all.
    multiprocess_mode="livesum",
)

# --- SSE streams ---
# Naive middleware records duration/in-progress at call_next return, which for a
# StreamingResponse is when headers are ready, not when the body finishes — so these are
# instrumented directly in the generator instead of via HTTP_DURATION/HTTP_IN_PROGRESS.
SSE_STREAMS_OPEN = Gauge(
    "sse_streams_open",
    "Currently open SSE streams",
    ["endpoint"],
    multiprocess_mode="livesum",  # same reasoning as HTTP_IN_PROGRESS
)
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
CHAT_ADMISSION_REJECTED = Counter(
    "chat_admission_rejected_total",
    "Chat requests refused with 503 because the queue was at CHAT_QUEUE_MAX_DEPTH",
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
CELERY_QUEUE = Gauge(
    "celery_queue_length",
    "Broker queue depth (waiting tasks only — excludes in-flight/unacked work)",
    ["queue_name"],
    # Absolute .set() of an external truth (Redis LLEN), not a delta, so it cannot leak
    # the way the in-flight gauges do. livemostrecent still beats the default "all":
    # it drops the pid label and, if a sampler is ever replaced, reports the newest
    # sample instead of exposing two competing series.
    multiprocess_mode="livemostrecent",
)
CELERY_TASKS_IN_FLIGHT = Gauge(
    "celery_tasks_in_flight",
    "Tasks currently executing (prerun -> postrun)",
    ["task_name"],
    # Incremented in prefork children, so the default per-PID series would need
    # summing by the query. livesum does it here and drops dead children's values —
    # but only children that reached worker_process_shutdown. A SIGKILLed child's +1
    # survives on disk, so correctness also depends on purge_multiproc_dir() running
    # at startup; see src/observability/multiproc.py.
    multiprocess_mode="livesum",
)

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

CHAT_STAGE_DURATION = Histogram(
    "chat_pipeline_stage_duration_seconds",
    "Chat pipeline stage latency",
    # stage: one of the _STAGE_LABELS keys in services/chat/tasks.py
    ["stage"],
    # agent_loop and stream_llm_response run tens of seconds; the rest are ms-scale.
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120),
)

# --- LLM cost/tokens/latency ---
LLM_TOKENS = Counter("llm_tokens_total", "Tokens", ["direction", "model"])
LLM_COST = Counter("llm_cost_usd_total", "Cost USD", ["model"])
LLM_CACHE_HIT_TOKENS = Counter("llm_cache_hit_tokens_total", "Cached input tokens", ["model"])
# request_type mirrors the llm_requests column: chat | chat_agent | agent_tool_call |
# router | rewrite_query | conversation_naming.
LLM_DURATION = Histogram(
    "llm_request_duration_seconds",
    "LLM call latency, first byte to last",
    ["model", "request_type"],
    buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 30, 60, 120),
)
LLM_TTFT = Histogram(
    "llm_time_to_first_token_seconds",
    "LLM time to first token (streaming calls only)",
    ["model", "request_type"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 10, 30),
)


def observe_llm_latency(model: str, request_type: str, stats: LLMResponseStats | None) -> None:
    """Record one LLM call's latency histograms."""
    if stats is None:
        return
    if stats.latency_ms is not None:
        LLM_DURATION.labels(model, request_type).observe(stats.latency_ms / 1000.0)
    if stats.ttft_ms is not None:
        LLM_TTFT.labels(model, request_type).observe(stats.ttft_ms / 1000.0)


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
    # Ceiling is 900 to clear DOCLING_PARSE_TIMEOUT_SECONDS=600 and the 1200s
    # Celery hard limit — a slow parse must land in a bucket, not in +Inf.
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600, 900),
)
INGESTION_QUEUE_WAIT = Histogram(
    "ingestion_queue_wait_seconds",
    "Upload -> task start",
    # Mirrors CHAT_QUEUE_WAIT. At one ingestion slot this is most of the
    # user-visible latency, so the buckets run out to the 1200s hard limit.
    # Measured from documents.created_at, which is stamped before the S3 PUT:
    # a large upload inflates this by the PUT's duration.
    buckets=(0.1, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600, 1200),
)
