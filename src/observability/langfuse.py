"""Langfuse observability client — cached per-process, no-op when disabled."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.utils.config import get_langfuse_config

if TYPE_CHECKING:
    from langfuse import Langfuse

logger = logging.getLogger(__name__)

_client: Langfuse | None = None
_enabled: bool = False


def get_client() -> Langfuse | None:
    """Return the cached Langfuse client, or None when disabled/unavailable."""
    return _client


def initialize() -> None:
    """Initialize the Langfuse client. Call once per process after config is loaded."""
    global _client, _enabled
    if _client is not None:
        return
    cfg = get_langfuse_config()
    _enabled = bool(cfg["enabled"])
    if not _enabled:
        logger.debug("langfuse.disabled")
        return

    try:
        from langfuse import Langfuse
    except ImportError:
        logger.warning("langfuse.import_failed", extra={"hint": "install langfuse>=3"})
        return

    _client = Langfuse(
        public_key=str(cfg["public_key"]),
        secret_key=str(cfg["secret_key"]),
        host=str(cfg["host"]),
        sample_rate=float(cfg["sample_rate"]),  # type: ignore[arg-type]
        environment=str(cfg["environment"]),
    )
    logger.info("langfuse.initialized", extra={"host": cfg["host"]})


def flush() -> None:
    """Flush pending events. Call in Celery task finally-block and FastAPI shutdown."""
    if _client is not None:
        _client.flush()


def reset() -> None:
    """Clear the cached client (call after fork, mirrors reset_client pattern)."""
    global _client, _enabled
    _client = None
    _enabled = False
