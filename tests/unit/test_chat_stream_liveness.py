"""The SSE subscriber must give up on a worker that died without saying so.

A worker killed by OOMKill, eviction or node loss never runs its exception handlers, so the
request's status stays non-terminal forever. The subscriber's only other give-up path is that
status, so without an elapsed-time rule it keeps the client on a heart-beating stream
indefinitely — measured at one full hour under load before this rule existed.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.api.routers.chat import _stream_event_time, _worker_gone
from src.utils.config import get_chat_stream_abandoned_after_seconds


def test_stream_id_parses_to_its_write_time() -> None:
    """Redis stream ids are `<unix-ms>-<seq>`, which is when the worker wrote the event."""
    written = datetime(2026, 9, 20, 16, 20, 22, tzinfo=UTC)
    event_id = f"{int(written.timestamp() * 1000)}-0"

    assert _stream_event_time(event_id) == written


def test_read_from_beginning_sentinel_is_not_a_timestamp() -> None:
    """`0-0` means "replay from the start", not 1970.

    Reading it as the epoch would make every request that has not yet emitted an event look
    abandoned by decades, and the subscriber would error out before the worker could start.
    """
    assert _stream_event_time("0-0") is None
    assert _stream_event_time("") is None
    assert _stream_event_time("not-an-id") is None


def test_slow_worker_is_not_declared_gone() -> None:
    """A task inside its hard time limit is slow, not dead, and must keep its stream open."""
    now = datetime(2026, 9, 20, 16, 30, 0, tzinfo=UTC)
    recent = now - timedelta(seconds=get_chat_stream_abandoned_after_seconds() - 30)

    assert _worker_gone("streaming", recent, now) is False


def test_silent_past_the_hard_limit_is_declared_gone() -> None:
    """Past the Celery hard limit plus a margin nothing can still be running, so the only
    explanation left is a worker that is gone."""
    now = datetime(2026, 9, 20, 17, 30, 0, tzinfo=UTC)
    stale = now - timedelta(seconds=get_chat_stream_abandoned_after_seconds() + 1)

    assert _worker_gone("streaming", stale, now) is True


@pytest.mark.parametrize("status", ["completed", "failed"])
def test_terminal_requests_are_never_declared_gone(status: str) -> None:
    """A finished request is finished however long ago it finished. Only the live statuses
    can be abandoned, or a reconnect to an old conversation would report a phantom failure."""
    now = datetime(2026, 9, 20, 17, 30, 0, tzinfo=UTC)
    ancient = now - timedelta(days=30)

    assert _worker_gone(status, ancient, now) is False


def test_naive_timestamps_do_not_raise() -> None:
    """`updated_at` is the fallback when a request has emitted no events yet. A driver that
    hands back a naive datetime must not take the whole stream down with a TypeError."""
    now = datetime(2026, 9, 20, 17, 30, 0, tzinfo=UTC)
    naive_stale = datetime(2026, 9, 20, 16, 20, 0)  # noqa: DTZ001 — the case under test

    assert _worker_gone("streaming", naive_stale, now) is True


def test_threshold_clears_the_worst_legitimate_service_time() -> None:
    """The margin has to sit above a real slow run, or the fix invents its own failures.

    T7 measured 94.1s as the worst service time under a uniformly slow LLM, against a 450s
    Celery hard limit. The default threshold must clear the hard limit, not merely that run.
    """
    assert get_chat_stream_abandoned_after_seconds() > 450
