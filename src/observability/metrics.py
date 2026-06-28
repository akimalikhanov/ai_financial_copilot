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

# --- Celery ---
CELERY_TASKS = Counter("celery_tasks_total", "Celery tasks", ["task_name", "state"])
CELERY_DURATION = Histogram("celery_task_duration_seconds", "Task duration", ["task_name"])
CELERY_QUEUE = Gauge("celery_queue_length", "Queue depth", ["queue_name"])

# --- Agentic RAG ---
RAG_RETRIEVAL = Histogram("rag_retrieval_duration_seconds", "Retrieval latency", ["stage"])
RAG_CHUNKS = Histogram("rag_chunks_retrieved", "Chunks per query", ["retriever"])
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
AGENT_TOOL_DURATION = Histogram("agent_tool_duration_seconds", "Tool latency", ["tool"])
ROUTER_DECISIONS = Counter("query_router_decisions_total", "Router decisions", ["decision"])
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
INGESTION_DURATION = Histogram(
    "ingestion_stage_duration_seconds",
    "Ingestion stage latency",
    ["stage"],  # parse | chunk | embed | upsert_qdrant | upsert_opensearch
)
