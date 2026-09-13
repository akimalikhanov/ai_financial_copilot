"""Tests for the T3 (SSE hold) leg of the load test's locustfile.

infra/loadtest/locustfile.py is not importable as `src.*` — it is a standalone file shipped
to the k8s Job as a ConfigMap — so it is loaded here by path.

Why these tests exist rather than trusting a manual run: T3's whole purpose is to catch the
API dropping idle SSE streams under load (docs/notes/loadtest-readiness-audit.md §4.1). A
`_hold_chat_stream` that reported a dropped stream as a successful hold would turn that
finding into a silent pass, which is worse than having no scenario at all — so the hold and
drop outcomes are pinned here in both directions. One earlier bug was of that family: the
deadline was checked only on line arrival, so a stream that went quiet blocked forever and
the hold never reported anything at all.

These use a real loopback HTTP server instead of respx: the behaviour under test is what
happens to a connection held open and then closed, which a mocked response object does not
model. The server must be threading — a single-threaded one serialises connections, so a
lingering hold from one case blocks the next and every later case reads as a false result.

**`locust` is never imported here.** `import locust` runs `gevent.monkey.patch_all()` at
module scope, and by the time pytest reaches this file the rest of the suite has already
imported `ssl` (via botocore/redis/aiohttp). Patching `ssl` after the fact corrupts
`SSLContext` and every later test dies with `RecursionError` — the whole suite, not just
this file, which is why running this module alone looks fine. So the locustfile is loaded
with a stub standing in for `locust`: the functions under test are plain httpx code that
never touches the framework, and the user classes only need `HttpUser`/`task`/`between` to
exist as names at class-creation time.
"""

from __future__ import annotations

import importlib.util
import sys
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

_LOCUSTFILE = Path(__file__).resolve().parents[2] / "infra" / "loadtest" / "locustfile.py"


def _locust_stub() -> types.ModuleType:
    """A stand-in for the `locust` package, structural only.

    `abstract` is the one attribute these tests actually read, and Locust's own semantics
    for it are plain class-attribute inheritance — a subclass that does not set it inherits
    the parent's value — so a bare base class reproduces the gating faithfully.
    """
    module = types.ModuleType("locust")

    class _Base:
        abstract = False
        host: str | None = None

    def _task(weight: Any = 1) -> Any:
        # Mirrors locust.task, which takes both @task and @task(weight) — the bare form
        # passes the function itself as `weight`.
        if callable(weight):
            return weight
        return lambda fn: fn

    def _between(_min: float, _max: float) -> Any:
        return lambda _self: 0.0

    module.HttpUser = _Base  # type: ignore[attr-defined]
    module.LoadTestShape = _Base  # type: ignore[attr-defined]
    module.task = _task  # type: ignore[attr-defined]
    module.between = _between  # type: ignore[attr-defined]
    return module


def _load_locustfile() -> Any:
    """Load the standalone locustfile by path with `locust` stubbed out (see module docstring).

    Registered in sys.modules so its module-level @dataclass can resolve its own __module__
    during class creation.
    """
    spec = importlib.util.spec_from_file_location("loadtest_locustfile", _LOCUSTFILE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    real_locust = sys.modules.get("locust")
    sys.modules["locust"] = _locust_stub()
    sys.modules["loadtest_locustfile"] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if real_locust is not None:
            sys.modules["locust"] = real_locust
        else:
            del sys.modules["locust"]
    return module


lf = _load_locustfile()


class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class _SseHandler(BaseHTTPRequestHandler):
    """Serves one of several SSE behaviours, selected per-request by the `request_id`."""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 — base class name
        pass  # keep pytest output clean

    def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler's required name
        # The mode rides in on `request_id`, not a query of its own: _hold_chat_stream builds
        # the URL as f"{base_url}/v1/chat/stream" and passes request_id through httpx params,
        # so a query appended to base_url would land mid-URL and never reach the handler.
        query = parse_qs(urlparse(self.path).query)
        mode = query.get("request_id", ["keepalive"])[0]

        if mode == "reject":
            self.send_response(404)
            self.end_headers()
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        try:
            if mode == "keepalive":
                # The handler alive with nothing to send — the state T3 measures.
                for _ in range(200):
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    time.sleep(0.05)
            elif mode == "silent":
                # Connection open but never writing again: the case a deadline check inside
                # the read loop could not escape.
                time.sleep(10)
            elif mode == "usage":
                self.wfile.write(b'id: 9-0\nevent: usage\ndata: {"persisted": true}\n\n')
                self.wfile.flush()
                time.sleep(10)
            elif mode == "error":
                self.wfile.write(b'id: 9-1\nevent: error\ndata: {"message": "boom"}\n\n')
                self.wfile.flush()
                time.sleep(10)
            elif mode == "close":
                self.wfile.write(b": keepalive\n\n")
                self.wfile.flush()
            elif mode == "keepalive_then_close":
                # Alive for ~1s, then drops. Straddles the deadline in both directions so a
                # short budget sees a healthy hold and a long one sees the drop.
                for _ in range(20):
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    time.sleep(0.05)
        except (BrokenPipeError, ConnectionResetError):
            pass


@pytest.fixture(scope="module")
def sse_server() -> Any:
    server = _ThreadingHTTPServer(("127.0.0.1", 0), _SseHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


def _hold(base: str, mode: str, hold_for_s: float) -> tuple[str, float, str]:
    """The mode is passed as the request_id — see _SseHandler.do_GET for why."""
    return lf._hold_chat_stream(base, mode, "tok", "4-0", hold_for_s)


class TestHoldChatStream:
    def test_keepalives_hold_until_deadline(self, sse_server: str) -> None:
        """A stream sending only `: keepalive` is a successful hold, and it lasts roughly
        the requested duration rather than returning early."""
        status, held_s, detail = _hold(sse_server, "keepalive", 1.0)
        assert status == "held", detail
        assert held_s >= 1.0

    def test_silent_stream_still_returns_at_deadline(self, sse_server: str) -> None:
        """Regression: with the deadline checked only on line arrival, a stream that goes
        quiet blocked forever and the hold never reported at all."""
        started = time.perf_counter()
        status, held_s, detail = _hold(sse_server, "silent", 1.0)
        assert status == "held", detail
        assert held_s >= 1.0
        assert time.perf_counter() - started < 5.0, "deadline did not fire on a silent stream"

    def test_server_closing_early_is_not_a_pass(self, sse_server: str) -> None:
        """A stream the server drops must never be reported as `held`."""
        status, held_s, detail = _hold(sse_server, "close", 5.0)
        assert status == "closed_early", f"dropped stream reported as {status}"
        assert "closed" in detail
        assert held_s < 5.0, "should report when the stream died, not when the budget expired"

    def test_budget_decides_hold_versus_drop(self, sse_server: str) -> None:
        """The same server behaviour must read as a pass or a failure depending only on
        whether the stream survived the whole budget. `keepalive_then_close` keepalives for
        ~1s then drops, so a short budget is a clean hold and a long one catches the drop —
        pinning the boundary between the two exits rather than either one alone."""
        status, held_s, detail = _hold(sse_server, "keepalive_then_close", 0.5)
        assert status == "held", (
            f"a stream alive through the whole budget is a pass, got {status}: {detail}"
        )
        assert held_s >= 0.5

        # Same server behaviour, but a budget that outlasts the close: now a failure.
        status, held_s, detail = _hold(sse_server, "keepalive_then_close", 5.0)
        assert status == "closed_early", f"dropped stream reported as {status}"
        assert held_s < 5.0

    @pytest.mark.parametrize("mode", ["usage", "error"])
    def test_terminal_events_end_the_hold(self, sse_server: str, mode: str) -> None:
        """A `usage`/`error` event means real work arrived on a stream T3 expects to be
        idle — reported as a failure so it cannot be mistaken for a clean hold."""
        status, _held_s, detail = _hold(sse_server, mode, 5.0)
        assert status == "closed_early"
        assert mode in detail

    def test_non_200_is_reported_as_http_error(self, sse_server: str) -> None:
        status, _held_s, detail = _hold(sse_server, "reject", 5.0)
        assert status == "http_error"
        assert "404" in detail


class TestModeGating:
    """Locust spawns every non-abstract HttpUser it finds, so exactly one profile must be
    active per run — mixing them would conflate T3's idle-connection measurement with chat
    throughput, which is the isolation the scenario depends on."""

    @pytest.mark.parametrize(
        ("mode", "expected"),
        [("ask", "ChatUser"), ("sse_hold", "SseHoldUser")],
    )
    def test_exactly_one_user_class_is_active(
        self, monkeypatch: pytest.MonkeyPatch, mode: str, expected: str
    ) -> None:
        monkeypatch.setenv("LOADTEST_MODE", mode)
        module = _load_locustfile()
        active = [
            name
            for name in ("ChatUser", "SseHoldUser", "_AuthenticatedUser")
            if not getattr(module, name).abstract
        ]
        assert active == [expected]

    def test_default_mode_is_the_asking_profile(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Every already-run scenario (T1/T2/T2b) depends on the unset default staying the
        question-asking profile."""
        monkeypatch.delenv("LOADTEST_MODE", raising=False)
        module = _load_locustfile()
        assert module.LOADTEST_MODE == "ask"
        assert not module.ChatUser.abstract
        assert module.SseHoldUser.abstract


class TestShapeGating:
    """Same hazard as the user classes: locust auto-registers every non-abstract
    LoadTestShape and silently picks the first, so at most one may be active per run."""

    @pytest.mark.parametrize(
        ("shape", "expected"),
        [("", []), ("ramp", ["RampShape"]), ("spike", ["SpikeShape"])],
    )
    def test_at_most_one_shape_is_active(
        self, monkeypatch: pytest.MonkeyPatch, shape: str, expected: list[str]
    ) -> None:
        monkeypatch.setenv("LOADTEST_SHAPE", shape)
        module = _load_locustfile()
        active = [
            name for name in ("RampShape", "SpikeShape") if not getattr(module, name).abstract
        ]
        assert active == expected

    def test_spike_holds_the_plateau_then_ends(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The whole reason T4 needs a shape: the hold must be measured from when the last
        user is up, not from when spawning started (which is what plain -t gives)."""
        monkeypatch.setenv("LOADTEST_SHAPE", "spike")
        monkeypatch.setenv("LOADTEST_SPIKE_USERS", "50")
        monkeypatch.setenv("LOADTEST_SPIKE_SPAWN_SECONDS", "10")
        monkeypatch.setenv("LOADTEST_SPIKE_HOLD", "2m")
        shape = _load_locustfile().SpikeShape()

        # Full target from tick 0 — the spike is a jump, not a climb; locust throttles the
        # actual arrival to spawn_rate (50/10s).
        assert shape.spawn_rate == 5.0
        for elapsed in (0.0, 5.0, 10.0, 129.0):
            shape.get_run_time = lambda _e=elapsed: _e
            assert shape.tick() == (50, 5.0)

        # Ends at spawn + hold, so the plateau really is 2 min long.
        shape.get_run_time = lambda: 130.0
        assert shape.tick() is None
