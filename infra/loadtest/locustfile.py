"""Locust load test for the chat and ingestion paths (Stage 17.5 Phase 8 / Stage 17 Phase 18).

Run locally against docker-compose (`make loadtest-local` / `make loadtest-local-ui`), or from
inside a kind/EKS cluster (`make k8s-loadtest`, see docs/notes/loadtest-concepts.md §8). This is
the one canonical locustfile both paths use — the k8s Job ships it as a ConfigMap, kept in sync
by `make k8s-sync-loadtest`, so edit here and never the copy under infra/k8s/loadtest/.

Why this isn't a plain HttpUser hitting one endpoint (see docs/notes/loadtest-concepts.md §2-3):
`POST /v1/chat` only enqueues and returns in milliseconds — the actual answer streams back
separately over SSE at `GET /v1/chat/stream`. One virtual user therefore has to model both calls
glued together, and the metric that matters is enqueue -> first token / enqueue -> usage, not the
POST's own latency. Locust's built-in client can't read `text/event-stream` incrementally, so the
SSE leg uses a plain `httpx.Client` and reports its own timings into Locust's stats via
`environment.events.request.fire()`.

`LOADTEST_MODE=upload` drives ingestion instead (docs/notes/loadtest-readiness-audit.md §8):
the same two-leg shape, POST /v1/documents/upload then the document's own SSE progress stream.
Note the FakeAdapter caveat applies differently there — it deletes the picture enricher's LLM
calls, so a fake ingestion run measures parse-bound time only (§8 item 1.5).

Point this at a stack whose LLM calls are routed to the FakeAdapter (MODELS_CONFIG_PATH ->
infra/config/models.loadtest.yaml) unless you mean to spend real money — see
docs/notes/loadtest-concepts.md §6.
"""

from __future__ import annotations

import itertools
import json
import os
import random
import time
import uuid
from dataclasses import dataclass

import httpx
from locust import HttpUser, LoadTestShape, between, task
from locust.exception import StopUser

# Must be a model id present in whichever models.yaml the target API is running with —
# infra/config/models.loadtest.yaml (fake) or infra/config/models.yaml (real, do not use
# without reading docs/notes/loadtest-concepts.md §6 first).
CHAT_MODEL = os.environ.get("LOADTEST_CHAT_MODEL", "gpt-4o-mini")

# Overall budget for one SSE read, matching the plan's soak/ramp scenarios where a slow
# answer is expected, not a bug. Measured p95 service time is ~101s (capacity-planning-
# concepts.md); leave headroom above that.
SSE_TIMEOUT_S = float(os.environ.get("LOADTEST_SSE_TIMEOUT_S", "180"))

# Think-time between questions from one user. Defaults match the measured mean (~219s); the
# T-series scenarios override these — a spike test wants near-zero, a soak wants realistic.
MIN_WAIT_S = float(os.environ.get("LOADTEST_MIN_WAIT", "180"))
MAX_WAIT_S = float(os.environ.get("LOADTEST_MAX_WAIT", "300"))

# Which user class generates load. "ask" (default) is the question-asking profile every
# throughput scenario uses (T1/T1b/T2/T2b/T4/T5/T6); "sse_hold" is T3, which opens a stream
# per user and holds it with no further questions; "upload" is the ingestion track (T11-T13),
# which uploads a PDF and follows it to a terminal status. Exactly one is active per run —
# Locust spawns every non-abstract HttpUser it finds, so mixing them would conflate T3's
# connection-holding measurement with chat throughput, which is the whole thing T3 isolates.
LOADTEST_MODE = os.environ.get("LOADTEST_MODE", "ask")

# T3 hold length. Defaults past any plausible -t so the run duration is the real bound and
# the hold does not end early on its own; set it below -t only to test reconnect churn.
HOLD_FOR_S = float(os.environ.get("LOADTEST_HOLD_FOR_S", "86400"))

# --- Ingestion (LOADTEST_MODE=upload), docs/notes/loadtest-readiness-audit.md §8 -------------

# Directory of PDFs to upload from, taken round-robin. Phase 1's three single-document classes
# live at /fixtures; item 2.2's 20-filing stratified sample lives at /fixtures/backlog. Which
# scenario a run exercises is therefore a matter of which directory it reads, not a code change.
FIXTURE_DIR = os.environ.get("LOADTEST_FIXTURE_DIR", "/fixtures")

# Restrict the pool to named files, e.g. "normal.pdf". Empty = everything in FIXTURE_DIR.
#
# The fixture classes differ by roughly 20x in service time (93-page text ~28.5s measured
# 2026-09-14, the same filing rasterized ~4 min at the OCR path's ~2.7s/page), so a run drawing
# from all three reports a variance that is mostly fixture mix and a mean describing no document
# that exists. Item 2.1 pins this to one class; T12 pins it to scanned.pdf to hold the GPU on the
# path it is measuring. Item 2.2 instead selects a whole directory via LOADTEST_FIXTURE_DIR.
FIXTURE_NAMES = [n.strip() for n in os.environ.get("LOADTEST_FIXTURES", "").split(",") if n.strip()]

# Deliberately far above the chat budget: ingestion is minutes, not seconds. It must sit above
# every server-side bound, or the client reports a timeout for a document the worker is still
# legitimately working on. Those bounds are DOCLING_PARSE_TIMEOUT_SECONDS=600 and the Celery
# limits — globally 450/360, which is BELOW the parse wrapper (audit §4.7, still open); the
# ingestion runs raise them per-Deployment via `make k8s-loadtest-ingest-limits`.
# T13 lowers the *server's* parse timeout instead (§8 item 1.4) rather than this.
INGEST_TIMEOUT_S = float(os.environ.get("LOADTEST_INGEST_TIMEOUT_S", "1500"))

# Think-time between uploads for one uploader. Defaults to zero — the backlog run (§8 item
# 2.2) wants N documents queued as fast as they can be posted, and at one ingestion slot the
# queue, not the client, is what paces the run.
UPLOAD_MIN_WAIT_S = float(os.environ.get("LOADTEST_UPLOAD_MIN_WAIT", "0"))
UPLOAD_MAX_WAIT_S = float(os.environ.get("LOADTEST_UPLOAD_MAX_WAIT", "0"))

# Uploads per user before it stops posting (it stays spawned, holding no work). §8 item 2.2
# is "a 20-document backlog", a fixed population — an open-ended upload loop would instead
# measure how fast the client can outrun one slot, which is not a number anyone needs.
UPLOADS_PER_USER = int(os.environ.get("LOADTEST_UPLOADS_PER_USER", "1"))

SAMPLE_QUESTIONS = [
    "What was total revenue last quarter?",
    "Summarize the key risk factors mentioned in the filing.",
    "What was the operating margin for the most recent fiscal year?",
    "How much cash and cash equivalents does the company hold?",
    "What were the main drivers of the change in net income?",
]


def _parse_duration_seconds(value: str) -> float:
    """Parse locust's -t syntax ("30m", "90s", "1h", bare "300" = seconds) for RampShape,
    which has to enforce the run duration itself — see the comment on run_time_seconds."""
    text = value.strip().lower()
    multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    if text and text[-1] in multipliers:
        return float(text[:-1]) * multipliers[text[-1]]
    return float(text)


@dataclass
class _StreamResult:
    status: str  # "usage" | "error" | "timeout" | "http_error"
    first_token_s: float | None
    usage_s: float | None
    detail: str = ""
    last_event_id: str | None = None


def _read_chat_stream(
    base_url: str, request_id: str, token: str, started_at: float
) -> _StreamResult:
    """Read the SSE stream to completion, timing first `delta` and terminal `usage` events.

    Hand-rolled rather than a library: the wire format is exactly what
    src/api/exceptions.py::_sse_event emits — optional `id: `, then `event: `, then `data: `,
    blocks separated by a blank line, plus `: ok` / `: keepalive` comment lines to skip.

    `last_event_id` on the result is the final `id:` seen — the Redis stream id T3's hold leg
    needs as its `after_event_id`, so its reconnect blocks on new events instead of replaying
    the terminal `usage` that would end the stream immediately.
    """
    first_token_s: float | None = None
    event_type: str | None = None
    data_line: str | None = None
    last_event_id: str | None = None

    try:
        with (
            httpx.Client(timeout=SSE_TIMEOUT_S) as client,
            client.stream(
                "GET",
                f"{base_url}/v1/chat/stream",
                params={"request_id": request_id},
                headers={"Authorization": f"Bearer {token}"},
            ) as response,
        ):
            if response.status_code != 200:
                return _StreamResult("http_error", None, None, f"status={response.status_code}")
            for line in response.iter_lines():
                if line == "":
                    event_type, data_line = None, None
                    continue
                if line.startswith(":"):
                    continue
                if line.startswith("id:"):
                    last_event_id = line[len("id:") :].strip()
                    continue
                if line.startswith("event:"):
                    event_type = line[len("event:") :].strip()
                    continue
                if line.startswith("data:"):
                    data_line = line[len("data:") :].strip()
                if event_type is None or data_line is None:
                    continue

                if event_type == "delta" and first_token_s is None:
                    first_token_s = time.perf_counter() - started_at

                if event_type == "usage":
                    try:
                        data = json.loads(data_line)
                    except json.JSONDecodeError:
                        data = {}
                    if data.get("persisted"):
                        return _StreamResult(
                            "usage",
                            first_token_s,
                            time.perf_counter() - started_at,
                            last_event_id=last_event_id,
                        )

                if event_type == "error":
                    return _StreamResult(
                        "error", first_token_s, None, data_line, last_event_id=last_event_id
                    )
    except httpx.TimeoutException:
        return _StreamResult("timeout", first_token_s, None, last_event_id=last_event_id)
    except httpx.HTTPError as exc:
        return _StreamResult(
            "http_error", first_token_s, None, str(exc), last_event_id=last_event_id
        )

    return _StreamResult(
        "error",
        first_token_s,
        None,
        "stream closed with no usage event",
        last_event_id=last_event_id,
    )


def _hold_chat_stream(
    base_url: str, request_id: str, token: str, after_event_id: str, hold_for_s: float
) -> tuple[str, float, str]:
    """T3's hold leg: keep one SSE connection open for `hold_for_s`, reading nothing new.

    Resuming at `after_event_id` (the last id the answer leg saw) is what makes this hold
    rather than return instantly. src/api/routers/chat.py::chat_stream_subscribe replays the
    Redis stream from the given id, and the terminal `usage` event ends the generator
    (chat.py:277-279) — so the default `0-0` would replay that same `usage` and close. Past
    it, `xread` has nothing to return and the handler sits in its 15s keepalive loop, which
    is exactly the "stream open, no work behind it" state §4.1 claims is cheap.

    Returns (status, held_seconds, detail). Only reaching `hold_for_s` is a pass; the server
    closing early, erroring, or emitting a real event is the interesting failure — a stream
    dropped under load is precisely the §4.1 counter-evidence this scenario exists to catch,
    so every non-`held` outcome must stay distinguishable from a pass.

    The deadline is a socket read timeout, not a clock check inside the read loop. A check
    inside the loop only runs when a line arrives, which breaks on exactly the stream this
    scenario cares about: with no read timeout, a handler that goes quiet blocks
    `iter_lines()` forever and the hold never reports at all — the run's own `-t` kills it
    and T3 produces no number. With a read deadline the quiet stream is what ends the hold,
    which is also the right semantics: the timeout firing at the deadline IS the successful
    hold, so `held` and `closed_early` come from structurally different exits rather than
    from comparing elapsed time against the budget.
    """
    started = time.perf_counter()

    def _elapsed() -> float:
        return time.perf_counter() - started

    try:
        with (
            # read = the remaining hold, so the socket unblocks exactly at the deadline
            # instead of waiting on a keepalive that may never come.
            httpx.Client(
                timeout=httpx.Timeout(connect=30.0, read=hold_for_s, write=30.0, pool=30.0)
            ) as client,
            client.stream(
                "GET",
                f"{base_url}/v1/chat/stream",
                params={"request_id": request_id, "after_event_id": after_event_id},
                headers={"Authorization": f"Bearer {token}"},
            ) as response,
        ):
            if response.status_code != 200:
                return ("http_error", _elapsed(), f"status={response.status_code}")
            for line in response.iter_lines():
                if _elapsed() >= hold_for_s:
                    return ("held", _elapsed(), "")
                # `: keepalive` every 15s is the expected traffic — the handler proving it is
                # alive with no events to send. Anything else means the hold ended for a
                # reason worth reporting.
                if line.startswith(":") or line == "":
                    continue
                if line.startswith("event:"):
                    event_type = line[len("event:") :].strip()
                    if event_type in ("error", "usage"):
                        return ("closed_early", _elapsed(), f"server sent {event_type}")
    except httpx.ReadTimeout:
        # The deadline fired with the connection still open and quiet: a successful hold.
        return ("held", _elapsed(), "")
    except httpx.TimeoutException:
        return ("timeout", _elapsed(), "connect/write timeout")
    except httpx.HTTPError as exc:
        return ("http_error", _elapsed(), str(exc))

    # iter_lines() ended on its own, meaning the server closed the connection. Always a
    # failure, unconditionally: a healthy hold exits via the ReadTimeout above, so reaching
    # here at all means the stream did not survive its budget.
    return ("closed_early", _elapsed(), "server closed the stream")


@dataclass
class _IngestResult:
    status: str  # "done" | "error" | "timeout" | "http_error"
    first_stage_s: float | None  # enqueue -> first stage event ~= queue wait at the client
    done_s: float | None
    last_stage: str = ""
    stage_count: int = 0
    detail: str = ""


def _read_ingestion_stream(
    base_url: str, document_id: str, token: str, started_at: float
) -> _IngestResult:
    """Follow one document's ingestion SSE stream to a terminal event.

    Same wire format as the chat stream, with three differences that matter here
    (src/api/routers/documents.py::ingestion_stream):
      * no `id:` lines — this stream has no resume, so there is nothing to track;
      * terminal events are `done` / `error` and carry no `persisted` flag, so the event
        type alone ends the read;
      * a document already terminal when the stream opens gets a synthetic `done`/`error`
        immediately (documents.py:257-276). That is a real outcome, not a short read: at
        one ingestion slot a fast document can finish before the client connects.

    `first_stage_s` is the client-side view of what §8 item 0.2 adds server-side as
    INGESTION_QUEUE_WAIT — the gap from upload to the worker's first `stage` event is
    queue wait plus startup. Reported separately from total because at one slot those are
    the two halves that move independently: service time is a property of the document,
    queue wait a property of the backlog ahead of it.
    """
    first_stage_s: float | None = None
    event_type: str | None = None
    data_line: str | None = None
    last_stage = ""
    stage_count = 0

    try:
        with (
            httpx.Client(timeout=INGEST_TIMEOUT_S) as client,
            client.stream(
                "GET",
                f"{base_url}/v1/documents/{document_id}/stream",
                headers={"Authorization": f"Bearer {token}"},
            ) as response,
        ):
            if response.status_code != 200:
                return _IngestResult(
                    "http_error", None, None, detail=f"status={response.status_code}"
                )
            for line in response.iter_lines():
                if line == "":
                    event_type, data_line = None, None
                    continue
                if line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    event_type = line[len("event:") :].strip()
                    continue
                if line.startswith("data:"):
                    data_line = line[len("data:") :].strip()
                if event_type is None or data_line is None:
                    continue

                try:
                    data = json.loads(data_line)
                except json.JSONDecodeError:
                    data = {}

                if event_type == "stage":
                    if first_stage_s is None:
                        first_stage_s = time.perf_counter() - started_at
                    stage_count += 1
                    last_stage = str(data.get("stage", ""))

                if event_type == "done":
                    return _IngestResult(
                        "done",
                        first_stage_s,
                        time.perf_counter() - started_at,
                        last_stage=last_stage,
                        stage_count=stage_count,
                    )

                if event_type == "error":
                    return _IngestResult(
                        "error",
                        first_stage_s,
                        None,
                        last_stage=last_stage,
                        stage_count=stage_count,
                        detail=str(data.get("message", ""))[:200],
                    )
    except httpx.TimeoutException:
        return _IngestResult(
            "timeout", first_stage_s, None, last_stage=last_stage, stage_count=stage_count
        )
    except httpx.HTTPError as exc:
        return _IngestResult(
            "http_error",
            first_stage_s,
            None,
            last_stage=last_stage,
            stage_count=stage_count,
            detail=str(exc),
        )

    # The stream ended without `done` or `error`. Not a pass: documents.py only closes the
    # generator on a terminal event or on a Redis read failure it deliberately does not
    # report as `error` (documents.py:340-348), so reaching here means the outcome is
    # genuinely unknown and must not be counted as a success.
    return _IngestResult(
        "error",
        first_stage_s,
        None,
        last_stage=last_stage,
        stage_count=stage_count,
        detail="stream closed with no terminal event",
    )


def _load_fixtures(directory: str, names: list[str]) -> list[str]:
    """Absolute paths of the PDFs available to upload.

    Read once at import rather than per task: the set is static for a run, and re-statting
    the directory on every upload would put client-side I/O inside the timed section.
    """
    if not os.path.isdir(directory):
        return []
    found = sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.lower().endswith(".pdf")
    )
    if not names:
        return found

    by_name = {os.path.basename(p): p for p in found}
    missing = [n for n in names if n not in by_name]
    if missing:
        # Loud, at import. A silently-dropped name would leave the run drawing from whatever
        # else the directory holds and report a service time for the wrong document class.
        raise RuntimeError(
            f"LOADTEST_FIXTURES names {missing} not in {directory!r} "
            f"(has: {sorted(by_name) or 'nothing'})"
        )
    return [by_name[n] for n in names]


FIXTURES = _load_fixtures(FIXTURE_DIR, FIXTURE_NAMES)

# Round-robin, NOT random.choice. Item 2.2's backlog is a stratified sample of 20 filings
# uploaded by 20 users — with independent random picks that covers only ~12.8 of the 20 on
# average and duplicates the rest, so the run would measure a random multiset rather than the
# distribution the sample was built to represent, and would miss the slowest document (the one
# that dominates E[S^2]) about a third of the time. Cycling makes "N users over an N-document
# set" mean exactly that, and is identical to random.choice when the pool holds one fixture.
#
# next() on a cycle has no yield point, so it is atomic across gevent greenlets.
_FIXTURE_CYCLE = itertools.cycle(FIXTURES) if FIXTURES else None


class RampShape(LoadTestShape):
    """T2 (docs/notes/loadtest-concepts.md §4/§7 step 5): ramp 1->60 askers, +2 every 30s,
    to find the knee — the concurrency where p95 queue wait stops being flat and starts
    climbing the ρ/(1-ρ) hyperbola. Only active when LOADTEST_SHAPE=ramp; the steady-state
    scenarios (T1, T1b, T3, T5, T6) don't set that and use plain -u/-r/-t instead, per the
    doc's step 4 ("steady scenarios don't need one").

    A LoadTestShape's tick() overrides -u/-r/-t entirely for the run, and Locust
    auto-registers *every* non-abstract LoadTestShape subclass it finds in the locustfile —
    with more than one defined, main.py just picks the first one it happens to encounter.
    `abstract` gates that: True means "not a real shape, skip me," so this only activates
    when LOADTEST_SHAPE=ramp is set. Every steady-state scenario (T1, T1b, T3, T5, T6) leaves
    it unset and gets plain -u/-r/-t behaviour, per the doc's step 4.
    """

    abstract = os.environ.get("LOADTEST_SHAPE") != "ramp"

    initial_users = int(os.environ.get("LOADTEST_RAMP_INITIAL_USERS", "1"))
    step_users = int(os.environ.get("LOADTEST_RAMP_STEP_USERS", "2"))
    step_seconds = float(os.environ.get("LOADTEST_RAMP_STEP_SECONDS", "30"))
    max_users = int(os.environ.get("LOADTEST_RAMP_MAX_USERS", "60"))
    # Within-step spawn rate: high enough that a step's new users are all present well before
    # the next step fires 30s later, so each step reads as a clean concurrency plateau rather
    # than a blurred ramp.
    spawn_rate = float(os.environ.get("LOADTEST_RAMP_SPAWN_RATE", "10"))
    # Locust IGNORES --run-time when a shape is active ("The following option(s) will be
    # ignored: --users, --spawn-rate, --run-time") — a shape runs until its own tick() returns
    # None. So the duration has to be read here, or LOADTEST_DURATION silently does nothing and
    # the ramp never ends. Parsed from the same string -t takes (e.g. "30m", "90s", "1h").
    run_time_seconds = _parse_duration_seconds(os.environ.get("LOADTEST_DURATION", "30m"))

    def tick(self) -> tuple[int, float] | None:
        run_time = self.get_run_time()
        if run_time >= self.run_time_seconds:
            return None  # ends the run, the shape's equivalent of -t elapsing
        step = int(run_time // self.step_seconds)
        user_count = min(self.initial_users + step * self.step_users, self.max_users)
        return (user_count, self.spawn_rate)


class SpikeShape(LoadTestShape):
    """T4 (docs/notes/loadtest-readiness-audit.md:688, concepts §7 step 6): jump 0->50 users
    in 10s, then hold at that plateau. Only active when LOADTEST_SHAPE=spike.

    A shape rather than plain `-u 50 -r 5` because -t starts counting when spawning starts,
    so the flat form gives ~1:50 of plateau for a nominal 2 min. Here the hold is measured
    from the moment the last user is up, which is the interval the pass bar is about.
    """

    abstract = os.environ.get("LOADTEST_SHAPE") != "spike"

    users = int(os.environ.get("LOADTEST_SPIKE_USERS", "50"))
    spawn_seconds = float(os.environ.get("LOADTEST_SPIKE_SPAWN_SECONDS", "10"))
    hold_seconds = _parse_duration_seconds(os.environ.get("LOADTEST_SPIKE_HOLD", "2m"))
    # Locust ignores --run-time under a shape (see RampShape.run_time_seconds), so the run
    # ends when spawn + hold elapses. LOADTEST_DURATION is deliberately unused here.
    spawn_rate = users / spawn_seconds if spawn_seconds > 0 else float(users)

    def tick(self) -> tuple[int, float] | None:
        if self.get_run_time() >= self.spawn_seconds + self.hold_seconds:
            return None
        return (self.users, self.spawn_rate)


class _AuthenticatedUser(HttpUser):
    """Shared setup for both profiles: a fresh synthetic account and one conversation.

    abstract = True keeps Locust from spawning this base itself; the two concrete profiles
    below each gate on LOADTEST_MODE so exactly one is active per run.
    """

    abstract = True

    def on_start(self) -> None:
        email = f"loadtest-{uuid.uuid4()}@example.com"
        password = "loadtest-password-not-real"  # noqa: S105 — synthetic account, throwaway
        resp = self.client.post(
            "/v1/auth/register",
            json={"email": email, "password": password},
            name="/v1/auth/register",
        )
        resp.raise_for_status()
        self.token = resp.json()["access_token"]

        resp = self.client.post(
            "/v1/conversations",
            json={"title": "Load test"},
            headers=self._auth_headers(),
            name="/v1/conversations",
        )
        resp.raise_for_status()
        self.conversation_id = resp.json()["conversation_id"]

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}


class ChatUser(_AuthenticatedUser):
    """Asks questions on one conversation for the life of the run — approximating one real
    user's session shape (measured mean think-time ~219s, see
    docs/notes/capacity-planning-concepts.md §1). The profile for every throughput scenario."""

    abstract = LOADTEST_MODE != "ask"

    wait_time = between(MIN_WAIT_S, MAX_WAIT_S)

    # T4's other half: "API /healthz stays under 1s" needs /healthz actually sampled during
    # the burst. Weighted 1 against ask_question's 10 to stay a probe rather than load — it
    # must not displace the questions that create the saturation being probed.
    @task(1)
    def health_check(self) -> None:
        self.client.get("/healthz", name="/healthz")

    @task(10)
    def ask_question(self) -> None:
        started_at = time.perf_counter()
        body = {
            "conversation_id": self.conversation_id,
            "content": random.choice(SAMPLE_QUESTIONS),
            "client_msg_id": str(uuid.uuid4()),
            "client_request_id": str(uuid.uuid4()),
            "model": CHAT_MODEL,
            "params": {},
            "metadata": {},
        }
        with self.client.post(
            "/v1/chat",
            json=body,
            headers=self._auth_headers(),
            name="/v1/chat (enqueue)",
            catch_response=True,
        ) as resp:
            if resp.status_code == 503:
                # Admission control shedding at CHAT_QUEUE_MAX_DEPTH. Still a question this
                # user got no answer to, so it stays a failure — but reported under its own
                # name so T4's "no 5xx" bar can be read as "no 5xx *other than* this".
                # Uncontrolled 5xx keep the generic name below.
                resp.failure(f"admission shed: 503 (Retry-After={resp.headers.get('Retry-After')})")
                return
            if resp.status_code != 200:
                resp.failure(f"enqueue failed: {resp.status_code} {resp.text[:200]}")
                return
            request_id = resp.json()["request_id"]

        assert self.host is not None, "run with --host, e.g. http://localhost:8000"
        result = _read_chat_stream(self.host, str(request_id), self.token, started_at)

        # self.environment.events, not the module-level `events`: in distributed mode the
        # module-level object is not the runner's, so stats fired on it never reach the master.
        events = self.environment.events

        if result.first_token_s is not None:
            events.request.fire(
                request_type="SSE",
                name="chat_first_token",
                response_time=result.first_token_s * 1000,
                response_length=0,
                exception=None,
            )

        total_s = time.perf_counter() - started_at
        if result.status == "usage":
            events.request.fire(
                request_type="SSE",
                name="chat_usage",
                response_time=(result.usage_s or total_s) * 1000,
                response_length=0,
                exception=None,
            )
        else:
            events.request.fire(
                request_type="SSE",
                name="chat_usage",
                response_time=total_s * 1000,
                response_length=0,
                exception=RuntimeError(f"{result.status}: {result.detail}"),
            )


class SseHoldUser(_AuthenticatedUser):
    """T3 (docs/notes/loadtest-readiness-audit.md:671, concepts §7 step 2): open one SSE
    stream per user and hold it, asking no further questions — to test §4.1's "SSE streams
    are cheap" claim against `pg_stat_activity`'s `idle in transaction` count and pgbouncer
    `cl_waiting`.

    Each user asks exactly ONE question at startup, because a stream needs a real
    request_id: chat.py:209-214 404s an unknown id, so ids cannot be fabricated. That one
    answer leg is startup cost, not the measurement — once it completes, the user reconnects
    past the last event id and holds. Steady state is therefore N idle streams and zero
    chat work, which is what makes the scenario attribute what it measures to the SSE leg
    alone.

    The hold is one task that runs for the whole run, so there is no wait_time: returning
    from it would mean the hold ended, and the `held` vs `closed_early` outcome fired below
    is the result. Drive concurrency with plain -u/-r/-t (no shape), per the doc's step 4.
    """

    abstract = LOADTEST_MODE != "sse_hold"

    def on_start(self) -> None:
        super().on_start()
        # The seed question. Its cost is paid once per user at spawn; the measurement is the
        # hold that follows, so this leg's latency is deliberately not reported into stats.
        body = {
            "conversation_id": self.conversation_id,
            "content": random.choice(SAMPLE_QUESTIONS),
            "client_msg_id": str(uuid.uuid4()),
            "client_request_id": str(uuid.uuid4()),
            "model": CHAT_MODEL,
            "params": {},
            "metadata": {},
        }
        resp = self.client.post(
            "/v1/chat",
            json=body,
            headers=self._auth_headers(),
            name="/v1/chat (T3 seed enqueue)",
        )
        resp.raise_for_status()
        self.request_id = str(resp.json()["request_id"])

        # Drain the answer to get the terminal event id. Without a real `after_event_id` the
        # hold would resume at 0-0, replay this `usage`, and close immediately.
        result = _read_chat_stream(str(self.host), self.request_id, self.token, time.perf_counter())
        self.after_event_id = result.last_event_id or "0-0"
        if result.last_event_id is None:
            # No id seen means the seed answer never landed; the hold would close instantly on
            # the replayed terminal event. Report it rather than logging a misleading pass.
            self.environment.events.request.fire(
                request_type="SSE",
                name="sse_seed_answer",
                response_time=0,
                response_length=0,
                exception=RuntimeError(f"seed answer did not complete: {result.status}"),
            )

    @task
    def hold_stream(self) -> None:
        status, held_s, detail = _hold_chat_stream(
            str(self.host), self.request_id, self.token, self.after_event_id, HOLD_FOR_S
        )
        self.environment.events.request.fire(
            request_type="SSE",
            name="sse_hold",
            response_time=held_s * 1000,
            response_length=0,
            exception=None if status == "held" else RuntimeError(f"{status}: {detail}"),
        )


class UploadUser(_AuthenticatedUser):
    """T11-T13 (docs/notes/loadtest-readiness-audit.md §8): upload a PDF and follow it to a
    terminal ingestion status.

    Same two-leg shape as ChatUser and for the same reason (concepts §2-3): POST
    /v1/documents/upload stores to S3 and enqueues, returning in well under a second, while
    the work itself runs in the ingestion worker and reports over SSE. The POST's latency is
    S3 upload time and says nothing about ingestion, so the numbers that matter —
    first_stage and done — are fired from the stream leg.

    Each user uploads UPLOADS_PER_USER documents and then idles. With one ingestion slot,
    concurrency here is a property of the *queue*, not of the client: -u 20 with the default
    one-upload-each is §8 item 2.2's 20-document backlog, posted near-simultaneously and
    drained serially by the worker.
    """

    abstract = LOADTEST_MODE != "upload"

    wait_time = between(UPLOAD_MIN_WAIT_S, UPLOAD_MAX_WAIT_S)

    def on_start(self) -> None:
        super().on_start()
        self._uploads_done = 0
        if not FIXTURES:
            # Nothing to upload — fail loudly at spawn rather than reporting a clean run in
            # which no document was ever ingested.
            raise RuntimeError(
                f"no PDFs in LOADTEST_FIXTURE_DIR={FIXTURE_DIR!r}; "
                "see docs/notes/loadtest-readiness-audit.md §8 items 1.2-1.3"
            )

    @task
    def upload_document(self) -> None:
        if self._uploads_done >= UPLOADS_PER_USER:
            # StopUser, not `return`: the upload wait_time is 0 (the backlog run wants its
            # documents posted as fast as they can be), so returning would put locust into a
            # tight loop of no-op task calls for the rest of -t. Measured at -u 1: a full core
            # at 97C, and at -u 20 those spinning greenlets would compete with the ones still
            # reading streams — inflating ingest_done for documents that are merely waiting.
            raise StopUser
        self._uploads_done += 1

        assert _FIXTURE_CYCLE is not None  # on_start raises if FIXTURES is empty
        path = next(_FIXTURE_CYCLE)
        with open(path, "rb") as fh:
            payload = fh.read()

        started_at = time.perf_counter()
        with self.client.post(
            "/v1/documents/upload",
            files={"file": (os.path.basename(path), payload, "application/pdf")},
            headers=self._auth_headers(),
            name="/v1/documents/upload (enqueue)",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"upload failed: {resp.status_code} {resp.text[:200]}")
                return
            # UploadDocumentResponse names the field `id`, not `document_id`
            # (src/api/routers/documents.py:170-177).
            document_id = str(resp.json()["id"])

        assert self.host is not None, "run with --host, e.g. http://localhost:8000"
        result = _read_ingestion_stream(self.host, document_id, self.token, started_at)

        # self.environment.events — see the note in ChatUser.ask_question.
        events = self.environment.events

        if result.first_stage_s is not None:
            events.request.fire(
                request_type="INGEST",
                name="ingest_first_stage",
                response_time=result.first_stage_s * 1000,
                response_length=0,
                exception=None,
            )

        total_s = time.perf_counter() - started_at
        events.request.fire(
            request_type="INGEST",
            name="ingest_done",
            response_time=(result.done_s or total_s) * 1000,
            response_length=0,
            exception=None
            if result.status == "done"
            else RuntimeError(
                f"{result.status} after {result.stage_count} stages "
                f"(last={result.last_stage or 'none'}): {result.detail}"
            ),
        )
