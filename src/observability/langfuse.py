"""Langfuse observability client — cached per-process, no-op when disabled."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING, Literal

from src.utils.config import get_langfuse_config

if TYPE_CHECKING:
    from langfuse import Langfuse

logger = logging.getLogger(__name__)

Level = Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"]

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
        logger.warning("langfuse.import_failed", extra={"hint": "install langfuse>=4"})
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


def describe_error(exc: BaseException) -> str:
    if isinstance(exc, asyncio.CancelledError):
        return "cancelled (caller timeout or shutdown)"
    return f"{type(exc).__name__}: {exc}"[:500]


def mark(obs: object, level: Level, message: str) -> None:
    """Set an observation's level. Langfuse never does this on its own: a failure that is
    caught (fallbacks, fail-open retrieval) or an exception leaving a span both stay
    DEFAULT unless marked here."""
    if obs is None:
        return
    with contextlib.suppress(Exception):
        obs.update(level=level, status_message=message)  # type: ignore[attr-defined]


def mark_current(level: Level, message: str) -> None:
    """`mark` for the current span, where the caller holds no observation object."""
    lf = get_client()
    if lf is None:
        return
    with contextlib.suppress(Exception):
        lf.update_current_span(level=level, status_message=message)


@contextlib.contextmanager
def span(
    name: str,
    *,
    as_type: str = "span",
    input: object = None,
    metadata: dict | None = None,
    **extra_metadata: object,
):
    """Start a Langfuse observation as the current span; no-op when disabled.

    Replaces the ExitStack + `if lf: ... enter_context(...) ... finally: stack.close()`
    boilerplate every call site otherwise reimplements — each copy was a place the
    instrumentation could (and did) silently drift out of sync with the state it traces.
    Yields the observation object, or None when Langfuse is disabled/unavailable — callers
    should guard `.update(...)` calls on that (`if obs: obs.update(...)`). An exception
    (or cancellation) leaving the block marks the observation ERROR.
    """
    lf = get_client()
    if lf is None:
        yield None
        return
    merged_metadata = {**(metadata or {}), **extra_metadata} or None
    with lf.start_as_current_observation(
        as_type=as_type,  # type: ignore[arg-type]
        name=name,
        input=input,
        metadata=merged_metadata,
    ) as obs:
        try:
            yield obs
        except BaseException as exc:
            mark(obs, "ERROR", describe_error(exc))
            raise
