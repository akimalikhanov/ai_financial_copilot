"""Deciding, from the subscriber's side, whether the worker behind an SSE stream is gone.

Both streaming endpoints read a Redis stream that a killed worker simply stops writing to,
and both face the same problem: a process that is OOMKilled, evicted or node-lost runs no
exception handler, so the row it was working on never reaches a terminal status and the
stream's only other give-up path never fires. The rule lives here rather than in either
router so the two paths cannot drift apart, and so it stays testable on its own.

The defaults are chat's, which is the caller that predates the split; the ingestion path
passes its own terminal statuses and its own deadline.
"""

from __future__ import annotations

from datetime import UTC, datetime

from src.utils.config import get_chat_stream_abandoned_after_seconds

# Statuses each worker sets on its own way out. Anything else means the pipeline is either
# still running or its worker died without running any handler.
CHAT_TERMINAL_STATUSES = frozenset({"completed", "failed"})
INGEST_TERMINAL_STATUSES = frozenset({"ready", "failed"})


def _stream_event_time(event_id: str) -> datetime | None:
    """Wall-clock time of a Redis stream id, which is `<unix-ms>-<seq>`.

    The id of the last event delivered is the most precise "when did this request last make
    progress" signal available to the subscriber — better than the request row, whose
    `updated_at` only moves on status transitions.
    """
    try:
        ms = int(event_id.split("-")[0])
    except (ValueError, IndexError):
        return None
    # "0-0" is the read-from-the-beginning sentinel, not a timestamp. Returning the epoch for
    # it would make every not-yet-started request look abandoned by ~56 years.
    if ms <= 0:
        return None
    try:
        return datetime.fromtimestamp(ms / 1000.0, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None


def _worker_gone(
    status: str | None,
    progress_at: datetime,
    now: datetime,
    terminal: frozenset[str] = CHAT_TERMINAL_STATUSES,
    after_seconds: float | None = None,
) -> bool:
    """True when a non-terminal request has made no progress for longer than a running task could.

    Split out from the subscriber so the rule is testable on its own: the subscriber supplies
    the two timestamps, this decides. `after_seconds` defaults to the chat deadline.
    """
    # `None` is not terminal: a request that never got a status is exactly the abandoned case.
    if status is not None and status in terminal:
        return False
    if progress_at.tzinfo is None:
        progress_at = progress_at.replace(tzinfo=UTC)
    if after_seconds is None:
        after_seconds = get_chat_stream_abandoned_after_seconds()
    return (now - progress_at).total_seconds() > after_seconds
