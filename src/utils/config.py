from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Load environment variables from .env file (if present)
load_dotenv()


def get_project_root() -> Path:
    """
    Get the project root directory.

    Uses PROJECT_ROOT environment variable if set, otherwise searches
    for pyproject.toml in parent directories.

    Returns:
        Path to the project root directory.

    Raises:
        RuntimeError: If project root cannot be determined.
    """
    if root := os.getenv("PROJECT_ROOT"):
        return Path(root)

    # Fallback: search for pyproject.toml
    current = Path(__file__).resolve()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent

    raise RuntimeError("Could not determine project root")


def _expand_env_vars(value: Any) -> Any:
    """Recursively expand ${VAR:-default} patterns in YAML values."""
    if isinstance(value, str):
        # Match ${VAR:-default} pattern
        pattern = r"\$\{([^:}]+)(?::-([^}]*))?\}"

        def replacer(match: re.Match[str]) -> str:
            var_name = match.group(1)
            default = match.group(2) if match.group(2) is not None else ""
            return os.getenv(var_name, default)

        return re.sub(pattern, replacer, value)
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    return value


def load_yaml_config(
    relative_path: str,
    *,
    config_path: Path | str | None = None,
    expand_env_vars: bool = False,
) -> dict[str, Any]:
    """
    Load a YAML config file from the project.

    Args:
        relative_path: Relative path from project root (e.g., "infra/config/models.yaml").
            Only used if config_path is None.
        config_path: Absolute path to config file. If provided, overrides relative_path.
        expand_env_vars: If True, expand ${VAR:-default} patterns in YAML values.

    Returns:
        Parsed YAML dict.
    """
    if config_path is None:
        project_root = get_project_root()
        config_path = project_root / relative_path
    else:
        config_path = Path(config_path)

    with open(config_path) as f:
        data = yaml.safe_load(f)

    if expand_env_vars:
        data = _expand_env_vars(data)

    return data


def load_models_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Load models.yaml config file with environment variable expansion.

    Args:
        config_path: Path to models.yaml. If None, uses infra/config/models.yaml relative to project root.

    Returns:
        Parsed YAML dict with env vars expanded.
    """
    if config_path is None:
        return load_yaml_config("infra/config/models.yaml", expand_env_vars=True)
    return load_yaml_config("", config_path=config_path, expand_env_vars=True)


def load_error_maps(config_path: Path | str | None = None) -> dict[int, dict[str, str]]:
    """
    Load error messages from error_maps.yaml config file.

    Args:
        config_path: Path to error_maps.yaml. If None, uses infra/config/error_maps.yaml
            relative to project root.

    Returns:
        Dictionary mapping status codes to dict with 'user' and 'internal' messages.
    """
    if config_path is None:
        data = load_yaml_config("infra/config/error_maps.yaml", expand_env_vars=False)
    else:
        data = load_yaml_config("", config_path=config_path, expand_env_vars=False)

    errors = data.get("errors", {})
    # Convert string keys to int keys
    return {int(k): v for k, v in errors.items()}


def get_cors_origins() -> list[str]:
    """
    Parse CORS_ALLOWED_ORIGINS from environment.

    CORS_ALLOWED_ORIGINS: Comma-separated list of allowed origins (required).
    Examples:
      - Development: "http://localhost:3000,http://127.0.0.1:3000"
      - Production:  "https://app.example.com,https://www.example.com"
      - Allow all (NOT recommended for production): "*"

    Raises:
        RuntimeError: If CORS_ALLOWED_ORIGINS is not set.
    """
    raw = os.getenv("CORS_ALLOWED_ORIGINS")
    if not raw:
        raise RuntimeError(
            "CORS_ALLOWED_ORIGINS environment variable is required. "
            "Set it to a comma-separated list of allowed origins "
            "(e.g., 'http://localhost:3000' for dev, or your production domain)."
        )
    if raw == "*":
        return ["*"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def get_db_url() -> str:
    """
    Build database connection URL from environment variables.

    Environment variables:
      - APP_DB_HOST: Database host (default: localhost)
      - APP_DB_PORT: Database port (default: 6432 for pgbouncer)
      - APP_DB_NAME: Database name (default: app)
      - APP_DB_USER: Database user (default: app)
      - APP_DB_PASSWORD: Database password (required)

    Returns:
        PostgreSQL async connection URL string.

    Raises:
        RuntimeError: If APP_DB_PASSWORD is not set.
    """
    host = os.getenv("APP_DB_HOST", "localhost")
    port = os.getenv("APP_DB_PORT", "6432")  # Default to pgbouncer port
    db_name = os.getenv("APP_DB_NAME", "app")
    db_user = os.getenv("APP_DB_USER", "app")
    db_password = os.getenv("APP_DB_PASSWORD")

    if not db_password:
        raise RuntimeError(
            "APP_DB_PASSWORD environment variable is required. "
            "Set it to the database password for the application database."
        )

    # Use asyncpg driver for async PostgreSQL connections
    return f"postgresql+asyncpg://{db_user}:{db_password}@{host}:{port}/{db_name}"


def get_rate_limit_window_ms() -> int:
    """Rate limit window in milliseconds (RATE_LIMIT_WINDOW_MS, default 60000)."""
    return int(os.getenv("RATE_LIMIT_WINDOW_MS", "60000"))


def get_rate_limit_max_requests() -> int:
    """Max requests per window (RATE_LIMIT_MAX_REQUESTS, default 30)."""
    return int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "30"))


def get_rate_limit_retry_after_sec() -> int:
    """Seconds to suggest in Retry-After header (derived from window)."""
    return get_rate_limit_window_ms() // 1000


def get_chat_tail_ttl() -> int:
    """Chat tail cache TTL in seconds (CHAT_TAIL_TTL, default 1800)."""
    return int(os.getenv("CHAT_TAIL_TTL", "1800"))


def get_chat_tail_max_messages() -> int:
    """Max messages in chat tail cache (CHAT_TAIL_MAX_MESSAGES, default 50)."""
    return int(os.getenv("CHAT_TAIL_MAX_MESSAGES", "50"))


# --- Redis ---
def get_chat_queue_stream() -> str:
    """Redis stream key for chat queue (CHAT_QUEUE_STREAM, default chat:queue)."""
    return os.getenv("CHAT_QUEUE_STREAM", "chat:queue")


def _build_redis_url(host: str, port: str, db: str, password: str | None) -> str:
    if password:
        return f"redis://:{password}@{host}:{port}/{db}"
    return f"redis://{host}:{port}/{db}"


def get_redis_app_url() -> str:
    """
    Redis for rate limit, cache, SSE stream (chat:events:*).
    Env: REDIS_APP_HOST, REDIS_APP_PORT, REDIS_APP_DB, REDIS_APP_PASSWORD.
    Falls back to REDIS_* if REDIS_APP_* not set.
    """
    host = os.getenv("REDIS_APP_HOST") or os.getenv("REDIS_HOST", "localhost")
    port = os.getenv("REDIS_APP_PORT") or os.getenv("REDIS_PORT", "6379")
    db = os.getenv("REDIS_APP_DB") or os.getenv("REDIS_DB", "0")
    password = os.getenv("REDIS_APP_PASSWORD") or os.getenv("REDIS_PASSWORD")
    return _build_redis_url(host, port, db, password)


def get_redis_broker_url() -> str:
    """
    Redis for Celery broker / queue (chat:queue, PDF tasks).
    Env: REDIS_BROKER_HOST, REDIS_BROKER_PORT, REDIS_BROKER_DB, REDIS_BROKER_PASSWORD.
    """
    host = os.getenv("REDIS_BROKER_HOST", "localhost")
    port = os.getenv("REDIS_BROKER_PORT", "6380")
    db = os.getenv("REDIS_BROKER_DB", "0")
    password = os.getenv("REDIS_BROKER_PASSWORD") or os.getenv("REDIS_PASSWORD")
    return _build_redis_url(host, port, db, password)


# --- S3 ---
def get_s3_endpoint_url() -> str:
    """S3/Garage endpoint (AWS_ENDPOINT_URL, default http://127.0.0.1:3900)."""
    return os.getenv("AWS_ENDPOINT_URL", "http://127.0.0.1:3900")


def get_s3_bucket() -> str:
    """Legacy S3/Garage bucket env (S3_BUCKET, default pdfs)."""
    return os.getenv("S3_BUCKET", "pdfs")


def get_s3_raw_bucket() -> str:
    """Bucket for raw PDFs (S3_RAW_BUCKET, falls back to S3_BUCKET, default pdfs)."""
    return os.getenv("S3_RAW_BUCKET") or get_s3_bucket()


def get_s3_docling_bucket() -> str:
    """Bucket for Docling JSON artifacts (S3_DOCLING_BUCKET, default docling)."""
    return os.getenv("S3_DOCLING_BUCKET", "docling")


def get_s3_rendered_bucket() -> str:
    """Bucket for rendered MD/HTML artifacts (S3_RENDERED_BUCKET, default rendered)."""
    return os.getenv("S3_RENDERED_BUCKET", "rendered")


def get_s3_chunks_bucket() -> str:
    """Bucket for chunks.jsonl artifacts (S3_CHUNKS_BUCKET, default chunks)."""
    return os.getenv("S3_CHUNKS_BUCKET", "chunks")


def get_s3_access_key() -> str:
    """S3/Garage access key (AWS_ACCESS_KEY_ID). Required for uploads."""
    val = os.getenv("AWS_ACCESS_KEY_ID")
    if not val:
        raise RuntimeError("AWS_ACCESS_KEY_ID is required for S3/Garage uploads")
    return val


def get_s3_secret_key() -> str:
    """S3/Garage secret key (AWS_SECRET_ACCESS_KEY). Required for uploads."""
    val = os.getenv("AWS_SECRET_ACCESS_KEY")
    if not val:
        raise RuntimeError("AWS_SECRET_ACCESS_KEY is required for S3/Garage uploads")
    return val


def _parse_bool(val: str | None, default: bool) -> bool:
    """Parse env var as bool. 1/true/yes/on -> True, 0/false/no/off -> False."""
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")


# --- Docling ---
def get_docling_do_ocr() -> bool:
    """DOCLING_DO_OCR (default: true)."""
    return _parse_bool(os.getenv("DOCLING_DO_OCR"), True)


def get_docling_do_table_structure() -> bool:
    """DOCLING_DO_TABLE_STRUCTURE (default: true)."""
    return _parse_bool(os.getenv("DOCLING_DO_TABLE_STRUCTURE"), True)


def get_docling_do_picture_description() -> bool:
    """DOCLING_DO_PICTURE_DESCRIPTION (default: true)."""
    return _parse_bool(os.getenv("DOCLING_DO_PICTURE_DESCRIPTION"), True)


def get_docling_generate_picture_images() -> bool:
    """DOCLING_GENERATE_PICTURE_IMAGES (default: false)."""
    return _parse_bool(os.getenv("DOCLING_GENERATE_PICTURE_IMAGES"), False)


def get_docling_generate_page_images() -> bool:
    """
    DOCLING_GENERATE_PAGE_IMAGES (default: false).
    When do_picture_description is true, this is forced to true (VLM needs page images).
    """
    if get_docling_do_picture_description():
        return True
    return _parse_bool(os.getenv("DOCLING_GENERATE_PAGE_IMAGES"), False)


def get_docling_picture_vlm_model() -> str:
    """DOCLING_PICTURE_VLM_MODEL — HuggingFace repo_id for picture description (default: SmolVLM-256M)."""
    return os.getenv("DOCLING_PICTURE_VLM_MODEL", "HuggingFaceTB/SmolVLM-256M-Instruct")


def get_docling_picture_vlm_prompt() -> str:
    """DOCLING_PICTURE_VLM_PROMPT — prompt sent to the VLM for each picture."""
    return os.getenv(
        "DOCLING_PICTURE_VLM_PROMPT",
        "Describe this financial chart or figure. "
        "Include: chart type, axes labels, key data points, trends, and notable patterns. "
        "Be precise with numbers and units.",
    )


def get_docling_document_timeout() -> float:
    """DOCLING_DOCUMENT_TIMEOUT in seconds (default: 300.0)."""
    val = os.getenv("DOCLING_DOCUMENT_TIMEOUT", "300")
    try:
        return float(val)
    except ValueError:
        return 300.0


def get_chunking_tokenizer_model() -> str:
    """CHUNKING_TOKENIZER_MODEL (default: nomic-ai/nomic-embed-text-v1.5)."""
    return os.getenv("CHUNKING_TOKENIZER_MODEL", "nomic-ai/nomic-embed-text-v1.5")


def get_chunking_max_tokens() -> int:
    """CHUNKING_MAX_TOKENS (default: 1000)."""
    val = os.getenv("CHUNKING_MAX_TOKENS", "1000")
    try:
        return int(val)
    except ValueError:
        return 1000


def get_chunking_min_tokens() -> int:
    """CHUNKING_MIN_TOKENS (default: 100)."""
    val = os.getenv("CHUNKING_MIN_TOKENS", "100")
    try:
        return int(val)
    except ValueError:
        return 100


def get_chunking_max_merge_multiplier() -> float:
    """CHUNKING_MAX_MERGE_MULTIPLIER (default: 2.0)."""
    val = os.getenv("CHUNKING_MAX_MERGE_MULTIPLIER", "2.0")
    try:
        return float(val)
    except ValueError:
        return 2.0


def get_embedding_provider() -> str:
    """EMBEDDING_PROVIDER (default: local). Use 'openai' for OpenAI API."""
    return os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()


def get_embedding_model() -> str:
    """EMBEDDING_MODEL or EMBED_MODEL_ID (default: all-MiniLM-L6-v2)."""
    return os.getenv("EMBEDDING_MODEL") or os.getenv("EMBED_MODEL_ID") or "all-MiniLM-L6-v2"


def get_embedding_dim() -> int | None:
    """EMBEDDING_DIM (optional). If set, validates embedding vector length."""
    raw = os.getenv("EMBEDDING_DIM")
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError("EMBEDDING_DIM must be an integer") from exc


# --- RAG retrieval ---
def get_query_router_model() -> str:
    """Model ID for query routing (QUERY_ROUTER_MODEL, default: gpt-4o-mini). Must exist in models.yaml."""
    return os.getenv("QUERY_ROUTER_MODEL", "gpt-4o-mini")


def get_rag_top_k() -> int:
    return int(os.getenv("RAG_TOP_K", "15"))


def get_rag_max_chunks() -> int:
    return int(os.getenv("RAG_MAX_CHUNKS", "10"))


def get_rag_max_tokens() -> int:
    return int(os.getenv("RAG_MAX_TOKENS", "4000"))


def get_rag_vector_weight() -> float:
    return float(os.getenv("RAG_VECTOR_WEIGHT", "0.6"))


def get_rag_score_threshold() -> float:
    return float(os.getenv("RAG_SCORE_THRESHOLD", "0.3"))


def get_vector_search_top_k() -> int:
    """VECTOR_SEARCH_TOP_K (default: 10)."""
    return int(os.getenv("VECTOR_SEARCH_TOP_K", "10"))


def get_keyword_search_top_k() -> int:
    """KEYWORD_SEARCH_TOP_K (default: 10)."""
    return int(os.getenv("KEYWORD_SEARCH_TOP_K", "10"))


def get_fuse_rrf_k() -> int:
    """FUSE_RRF_K (default: 60)."""
    return int(os.getenv("FUSE_RRF_K", "60"))


def get_fuse_rrf_final_top_k() -> int:
    """FUSE_RRF_FINAL_TOP_K (default: 15)."""
    return int(os.getenv("FUSE_RRF_FINAL_TOP_K", "15"))


def get_fuse_rrf_vector_weight() -> float:
    """FUSE_RRF_VECTOR_WEIGHT (default: 0.6)."""
    return float(os.getenv("FUSE_RRF_VECTOR_WEIGHT", "0.6"))


def get_fuse_rrf_keyword_weight() -> float:
    """FUSE_RRF_KEYWORD_WEIGHT (default: 0.4)."""
    return float(os.getenv("FUSE_RRF_KEYWORD_WEIGHT", "0.4"))


# --- Reranker ---
def get_reranker_enabled() -> bool:
    """RERANKER_ENABLED (default: False)."""
    return os.getenv("RERANKER_ENABLED", "false").lower() in ("true", "1", "yes")


def get_reranker_base_url() -> str:
    """RERANKER_BASE_URL (default: http://localhost:8080)."""
    return os.getenv("RERANKER_BASE_URL", "http://localhost:8080")


def get_reranker_model_name() -> str:
    """RERANKER_MODEL_NAME (default: Alibaba-NLP/gte-reranker-modernbert-base)."""
    return os.getenv("RERANKER_MODEL_NAME", "Alibaba-NLP/gte-reranker-modernbert-base")


def get_reranker_timeout_seconds() -> float:
    """RERANKER_TIMEOUT_SECONDS (default: 10.0)."""
    return float(os.getenv("RERANKER_TIMEOUT_SECONDS", "10.0"))


def get_reranker_top_k() -> int:
    """RERANKER_TOP_K (default: 10). Number of chunks to return after reranking."""
    return int(os.getenv("RERANKER_TOP_K", "10"))


def get_reranker_max_input() -> int:
    """RERANKER_MAX_INPUT (default: 30). Max chunks to send to reranker per query."""
    return int(os.getenv("RERANKER_MAX_INPUT", "30"))


def get_system_prompt_version() -> str:
    """SYSTEM_PROMPT_VERSION (default: v2). Controls which system prompt YAML is loaded."""
    return os.getenv("SYSTEM_PROMPT_VERSION", "v2")


def get_chat_retrieval_timeout() -> float:
    """CHAT_RETRIEVAL_TIMEOUT in seconds (default: 200.0). Per-backend retrieval timeout, fail-open."""
    val = os.getenv("CHAT_RETRIEVAL_TIMEOUT", "200")
    try:
        return float(val)
    except ValueError:
        return 200.0


# --- Query router ---
def get_router_config() -> dict[str, float | int]:
    """Query router configuration from environment variables.

    Returns:
        Dict with keys: temperature, max_tokens, entity_similarity_threshold,
        entity_max_candidates, filtered_md_thresh.
    """
    return {
        "temperature": float(os.getenv("ROUTER_TEMPERATURE", "0.0")),
        "max_tokens": int(os.getenv("ROUTER_MAX_TOKENS", "800")),
        "entity_similarity_threshold": float(os.getenv("ENTITY_SIMILARITY_THRESHOLD", "0.3")),
        "entity_max_candidates": int(os.getenv("ENTITY_MAX_CANDIDATES", "20")),
        "filtered_md_thresh": int(os.getenv("FILTERED_MD_THRESH", "5")),
    }
