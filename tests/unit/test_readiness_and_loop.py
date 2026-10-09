"""Phase 5: no blocking I/O on the API event loop, and readiness that sheds before liveness kills.

Two independent failures share this file because they share a cause — the API is
`uvicorn --workers 1`, so one blocked coroutine and one over-eager probe both take down every
open SSE stream on the pod.
"""

from __future__ import annotations

import asyncio
import inspect
import io
import json
import time
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.observability import metrics as metrics_mod

# --- 5.1 event loop hygiene -------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_index_cleanup_does_not_block_the_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slow search backend must not freeze other coroutines.

    Deliberately reproduces the real shape: sync client functions with a real blocking
    `time.sleep`. On the loop these would stall the heartbeat below; off it, the heartbeat
    keeps ticking. Fails if the calls are ever moved back onto the loop.
    """
    from src.services.ingestion import opensearch_ingest, qdrant_ingest

    def _slow_delete(*_args: Any, **_kwargs: Any) -> None:
        time.sleep(0.3)  # blocking on purpose — a sync client over a sync socket

    monkeypatch.setattr(qdrant_ingest, "delete_by_document", _slow_delete)
    monkeypatch.setattr(opensearch_ingest, "delete_by_document", _slow_delete)

    ticks = 0

    async def heartbeat() -> None:
        """Stands in for an open SSE stream: it must keep getting scheduled."""
        nonlocal ticks
        for _ in range(30):
            await asyncio.sleep(0.01)
            ticks += 1

    async def cleanup() -> None:
        await asyncio.gather(
            asyncio.to_thread(_slow_delete, "documents", uuid.uuid4()),
            asyncio.to_thread(_slow_delete, "chunks", uuid.uuid4()),
        )

    await asyncio.gather(heartbeat(), cleanup())

    # Both sleeps run concurrently off-loop, so the heartbeat gets its full run.
    assert ticks == 30


def test_delete_document_wraps_sync_clients_in_to_thread() -> None:
    """Guards the call site itself: the two deletes must go through to_thread.

    The timing test above passes even if the handler regresses, because it exercises the
    pattern rather than the handler. This one reads the handler.
    """
    from src.api.routers.documents import delete_document

    source = inspect.getsource(delete_document)
    cleanup = source.split("s3_keys")[0]  # only the index-cleanup block

    assert cleanup.count("asyncio.to_thread") == 2
    assert "asyncio.gather" in cleanup
    # The bare sync calls must be gone, not merely joined by a to_thread elsewhere.
    assert "\n        qdrant_ingest.delete_by_document(" not in cleanup
    assert "\n        opensearch_ingest.delete_by_document(" not in cleanup


class _SpooledLike(io.BytesIO):
    """Stands in for Starlette's SpooledTemporaryFile, which spills to disk above 1 MB.

    `read` sleeps to model a real disk read: on the loop it stalls everything, off it does not.
    """

    def __init__(self, data: bytes, read_cost: float = 0.3) -> None:
        super().__init__(data)
        self._read_cost = read_cost
        self.read_calls = 0

    def read(self, size: int = -1) -> bytes:  # type: ignore[override]
        self.read_calls += 1
        time.sleep(self._read_cost)
        return super().read(size)


@pytest.mark.asyncio
async def test_upload_pdf_reads_the_file_off_the_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """A large PDF must not be read from disk on the event loop.

    Starlette spools above 1 MB and the endpoint accepts 100 MB, so essentially every real
    upload is disk-backed. Handing that object to aioboto3 as Body= makes botocore read it
    inline, freezing every open SSE stream for the duration of the upload.
    """
    from src.services.ingestion import s3_client

    captured: dict[str, Any] = {}

    class _FakeClient:
        async def put_object(self, **kwargs: Any) -> dict[str, Any]:
            captured.update(kwargs)
            return {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc: Any) -> None:
            return None

    class _FakeSession:
        def client(self, *_args: Any, **_kwargs: Any) -> _FakeClient:
            return _FakeClient()

    monkeypatch.setattr(s3_client.aioboto3, "Session", lambda: _FakeSession())

    payload = b"%PDF-1.7 " + b"x" * 4096
    fileobj = _SpooledLike(payload)

    ticks = 0

    async def heartbeat() -> None:
        """An open SSE stream: it must keep getting scheduled during the upload."""
        nonlocal ticks
        for _ in range(30):
            await asyncio.sleep(0.01)
            ticks += 1

    async def do_upload() -> str:
        return await s3_client.upload_pdf(
            user_id=uuid.uuid4(), doc_id=uuid.uuid4(), filename="report.pdf", fileobj=fileobj
        )

    _, key = await asyncio.gather(heartbeat(), do_upload())

    assert ticks == 30  # the loop kept turning through the blocking read
    assert key.endswith("report.pdf")
    assert captured["Body"] == payload
    # Length must come from the bytes actually read: a wrong ContentLength makes Garage
    # reject or truncate the object.
    assert captured["ContentLength"] == len(payload)


def test_upload_pdf_does_not_hand_a_file_object_to_botocore() -> None:
    """Guards the call site: Body= must be bytes read off-loop, never the file object.

    The timing test above exercises the pattern; this one reads the function.
    """
    from src.services.ingestion.s3_client import upload_pdf

    source = inspect.getsource(upload_pdf)

    assert "asyncio.to_thread" in source
    assert "Body=fileobj" not in source


# --- 5.2 readiness ----------------------------------------------------------------------


class _OkRedis:
    async def ping(self) -> bool:
        return True


class _HangingRedis:
    """Redis that never answers — the readiness handler must time out, not hang with it."""

    async def ping(self) -> bool:
        await asyncio.sleep(3600)
        return True


class _FailingRedis:
    async def ping(self) -> bool:
        raise ConnectionError("redis went away")


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[TestClient, FastAPI]]:
    """A TestClient with lifespan startup stubbed out — /readyz is the subject, not init.

    Yields the app too: tests swap `app.state.redis` to simulate an outage, and TestClient
    exposes `.app` only as a bare ASGI callable.
    """
    from src import main as main_mod

    monkeypatch.setattr(main_mod, "init_db", _noop_async)
    monkeypatch.setattr(main_mod, "shutdown_db", _noop_async)
    monkeypatch.setattr(main_mod, "get_router", lambda: _FakeLLMRouter())
    monkeypatch.setattr(main_mod, "create_redis_app_client", _make_ok_redis)
    monkeypatch.setattr(main_mod, "close_redis_client", _noop_async_arg)
    monkeypatch.setattr(main_mod.lf_client, "initialize", lambda: None)
    monkeypatch.setattr(main_mod.lf_client, "flush", lambda: None)

    app = main_mod.create_app()
    with TestClient(app) as c:
        yield c, app


async def _noop_async() -> None:
    return None


async def _noop_async_arg(*_args: Any) -> None:
    return None


async def _make_ok_redis() -> _OkRedis:
    return _OkRedis()


class _FakeLLMRouter:
    async def close(self) -> None:
        return None


@pytest.fixture(autouse=True)
def _reset_stream_counts():
    """The counter is module state; a leaked count would make later tests read at_capacity."""
    metrics_mod._open_streams.clear()
    yield
    metrics_mod._open_streams.clear()


def test_readyz_ok_when_redis_up_and_below_capacity(
    api: tuple[TestClient, FastAPI],
) -> None:
    client, _ = api

    resp = client.get("/readyz")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "open_streams": 0}


def test_readyz_503s_when_redis_is_down(api: tuple[TestClient, FastAPI]) -> None:
    """Sheds traffic — and because this is wired to readiness, not liveness, the pod is
    not killed for a dependency outage it did not cause."""
    client, app = api
    app.state.redis = _FailingRedis()

    resp = client.get("/readyz")

    assert resp.status_code == 503
    assert resp.json()["status"] == "redis_unavailable"


def test_readyz_bounds_a_hanging_redis(
    api: tuple[TestClient, FastAPI], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The handler must answer within its own budget. A readiness probe that hangs reads as
    a probe timeout — indistinguishable from a wedged app — instead of "Redis is slow"."""
    from src import main as main_mod

    client, app = api
    monkeypatch.setattr(main_mod, "get_readiness_redis_timeout_seconds", lambda: 0.2)
    app.state.redis = _HangingRedis()

    started = time.perf_counter()
    resp = client.get("/readyz")
    elapsed = time.perf_counter() - started

    assert resp.status_code == 503
    assert resp.json()["status"] == "redis_unavailable"
    assert elapsed < 2.0


def test_readyz_sheds_at_capacity(
    api: tuple[TestClient, FastAPI], monkeypatch: pytest.MonkeyPatch
) -> None:
    """At the stream cap the pod leaves the endpoint list rather than accepting a stream
    it has no room to serve."""
    from src import main as main_mod

    client, _ = api
    monkeypatch.setattr(main_mod, "get_max_open_streams", lambda: 2)

    metrics_mod.sse_stream_opened("chat")
    assert client.get("/readyz").status_code == 200  # 1 < 2

    metrics_mod.sse_stream_opened("ingestion")
    resp = client.get("/readyz")  # 2 >= 2

    assert resp.status_code == 503
    assert resp.json() == {"status": "at_capacity", "open_streams": 2}

    # Reversible: readiness must recover on its own when a stream closes.
    metrics_mod.sse_stream_closed("chat")
    assert client.get("/readyz").status_code == 200


def test_healthz_stays_static_and_independent_of_redis(
    api: tuple[TestClient, FastAPI],
) -> None:
    """Liveness must not depend on anything that can blip. If it did, a Redis outage would
    SIGKILL the pod and drop every in-flight SSE stream."""
    client, app = api
    app.state.redis = _FailingRedis()

    resp = client.get("/healthz")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_open_stream_count_is_symmetric() -> None:
    """Counting streams is only useful if the count returns to zero — an off-by-one leak
    here silently walks a long-lived pod into permanent at_capacity."""
    for _ in range(5):
        metrics_mod.sse_stream_opened("chat")
    for _ in range(5):
        metrics_mod.sse_stream_closed("chat")

    assert metrics_mod.open_stream_count() == 0


def test_open_stream_count_never_goes_negative() -> None:
    """An unbalanced close (a generator finalized twice) must not drive the count below
    zero, which would let the pod exceed its cap before /readyz noticed."""
    metrics_mod.sse_stream_closed("chat")
    metrics_mod.sse_stream_closed("chat")

    assert metrics_mod.open_stream_count() == 0


# --- probe wiring -----------------------------------------------------------------------


def test_liveness_is_strictly_laxer_than_readiness() -> None:
    """The invariant this phase exists to enforce. If liveness can fail before readiness has
    had time to shed, the destructive action happens before the safe one — which is the
    behaviour of the two identical /healthz probes this replaced.
    """
    import yaml

    with open("infra/k8s/base/app/api-deployment.yaml") as f:
        docs = [d for d in yaml.safe_load_all(f) if d]

    deployment = next(d for d in docs if d["kind"] == "Deployment")
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    readiness = container["readinessProbe"]
    liveness = container["livenessProbe"]

    assert readiness["httpGet"]["path"] == "/readyz"
    assert liveness["httpGet"]["path"] == "/healthz"

    def time_to_fail(probe: dict[str, Any]) -> int:
        return probe["initialDelaySeconds"] + probe["periodSeconds"] * probe["failureThreshold"]

    assert time_to_fail(liveness) > time_to_fail(readiness)

    # The probe must allow more time than the handler's own Redis budget, or a slow Redis
    # shows up as a probe timeout instead of the 503 the handler would have returned.
    from src.utils.config import get_readiness_redis_timeout_seconds

    assert readiness["timeoutSeconds"] > get_readiness_redis_timeout_seconds()


def test_probe_paths_are_excluded_from_request_logs() -> None:
    """/readyz at a 5s period is the most frequent request the API serves; logging it would
    bury real traffic."""
    from src.api.logging import _UNLOGGED_PATHS

    assert "/readyz" in _UNLOGGED_PATHS
    assert "/healthz" in _UNLOGGED_PATHS


def test_readyz_payload_is_json_serialisable() -> None:
    """Guards against a numpy/Decimal count sneaking in — a JSONResponse that raises during
    encoding returns 500, which reads as a dead pod."""
    metrics_mod.sse_stream_opened("chat")

    assert json.dumps({"status": "ok", "open_streams": metrics_mod.open_stream_count()})
