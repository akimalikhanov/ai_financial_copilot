"""Unit tests for the TEI embedding path (Phase 7: batched, concurrent)."""

from __future__ import annotations

import json
import threading

import httpx
import pytest
import respx

from src.services.ingestion import embedder

TEI_URL = "http://tei-test:80"


@pytest.fixture(autouse=True)
def _tei_env(monkeypatch: pytest.MonkeyPatch):
    """Point the embedder at a mocked TEI and clear the per-process batch-size cache."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "tei")
    monkeypatch.setenv("EMBEDDER_BASE_URL", TEI_URL)
    monkeypatch.delenv("EMBEDDING_DIM", raising=False)
    embedder.reset_clients()
    yield
    embedder.reset_clients()


def _info_route(max_client_batch_size: int = 64) -> None:
    respx.get(f"{TEI_URL}/info").mock(
        return_value=httpx.Response(200, json={"max_client_batch_size": max_client_batch_size})
    )


def _embed_echo(request: httpx.Request) -> httpx.Response:
    """Return one vector per input, encoding the input's own value so order is checkable."""
    payload = json.loads(request.content)
    return httpx.Response(200, json=[[float(text)] for text in payload["inputs"]])


class TestBatchSizeResolution:
    @respx.mock
    def test_clamps_to_server_max(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A config larger than TEI's limit self-corrects instead of producing 413s."""
        monkeypatch.setenv("EMBEDDER_BATCH_SIZE", "512")
        _info_route(max_client_batch_size=32)
        assert embedder._resolve_tei_batch_size(TEI_URL, 5.0) == 32

    @respx.mock
    def test_keeps_configured_value_when_under_server_max(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EMBEDDER_BATCH_SIZE", "16")
        _info_route(max_client_batch_size=64)
        assert embedder._resolve_tei_batch_size(TEI_URL, 5.0) == 16

    @respx.mock
    def test_falls_back_to_config_when_info_unreachable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EMBEDDER_BATCH_SIZE", "8")
        respx.get(f"{TEI_URL}/info").mock(side_effect=httpx.ConnectError("down"))
        assert embedder._resolve_tei_batch_size(TEI_URL, 5.0) == 8

    @respx.mock
    def test_probes_once_per_process(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EMBEDDER_BATCH_SIZE", "64")
        route = respx.get(f"{TEI_URL}/info").mock(
            return_value=httpx.Response(200, json={"max_client_batch_size": 64})
        )
        embedder._resolve_tei_batch_size(TEI_URL, 5.0)
        embedder._resolve_tei_batch_size(TEI_URL, 5.0)
        assert route.call_count == 1


class TestEmbedTei:
    @respx.mock
    def test_vectors_stay_aligned_with_chunks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Concurrent batches must be reassembled in input order, not completion order."""
        monkeypatch.setenv("EMBEDDER_BATCH_SIZE", "2")
        monkeypatch.setenv("EMBEDDER_CONCURRENCY", "4")
        _info_route()
        respx.post(f"{TEI_URL}/embed").mock(side_effect=_embed_echo)

        chunks = [str(i) for i in range(9)]
        assert embedder.embed_chunks(chunks) == [[float(i)] for i in range(9)]

    @respx.mock
    def test_splits_into_batches_of_resolved_size(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EMBEDDER_BATCH_SIZE", "4")
        monkeypatch.setenv("EMBEDDER_CONCURRENCY", "2")
        _info_route()
        route = respx.post(f"{TEI_URL}/embed").mock(side_effect=_embed_echo)

        embedder.embed_chunks([str(i) for i in range(10)])
        assert route.call_count == 3  # 4 + 4 + 2

    @respx.mock
    def test_requests_overlap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The point of the phase: batches are in flight together, not one at a time."""
        monkeypatch.setenv("EMBEDDER_BATCH_SIZE", "1")
        monkeypatch.setenv("EMBEDDER_CONCURRENCY", "4")
        _info_route()

        in_flight = 0
        peak = 0
        lock = threading.Lock()
        barrier = threading.Barrier(4, timeout=5)

        def _slow(request: httpx.Request) -> httpx.Response:
            nonlocal in_flight, peak
            with lock:
                in_flight += 1
                peak = max(peak, in_flight)
            barrier.wait()
            with lock:
                in_flight -= 1
            return _embed_echo(request)

        respx.post(f"{TEI_URL}/embed").mock(side_effect=_slow)

        embedder.embed_chunks([str(i) for i in range(4)])
        assert peak == 4

    @respx.mock
    def test_batch_failure_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EMBEDDER_BATCH_SIZE", "1")
        monkeypatch.setenv("EMBEDDER_CONCURRENCY", "2")
        _info_route()
        respx.post(f"{TEI_URL}/embed").mock(return_value=httpx.Response(500))

        with pytest.raises(httpx.HTTPStatusError):
            embedder.embed_chunks(["a", "b"])

    def test_empty_input_short_circuits(self) -> None:
        assert embedder.embed_chunks([]) == []
