"""S3/Garage client for PDF uploads."""

from __future__ import annotations

import asyncio
import contextlib
import re
import tempfile
from pathlib import Path
from typing import Any
from uuid import UUID

import aioboto3
from botocore.config import Config

from src.utils.config import (
    get_s3_access_key,
    get_s3_bucket,
    get_s3_endpoint_url,
    get_s3_pictures_bucket,
    get_s3_raw_bucket,
    get_s3_secret_key,
)

# Of the 13 recorded ingestion failures in the loadtest audit, all were S3 connection
# timeouts — botocore's default retry mode doesn't reliably cover those. "standard" mode
# retries on connect/read timeouts and connection errors, not just throttling/5xx.
_CLIENT_CONFIG = Config(
    response_checksum_validation="when_required",
    retries={"max_attempts": 5, "mode": "standard"},
)
# The shared worker client carries the crop-upload fan-out plus the artifact uploads (whose
# multipart transfers open their own parallel parts), so botocore's default pool of 10 is short.
_WORKER_CLIENT_CONFIG = _CLIENT_CONFIG.merge(Config(max_pool_connections=32))

# One client per worker process, bound to the loop it was opened on. Constructing a Session and
# client per call meant a 300-picture document paid 300 client setups, pools and handshakes.
_client: Any | None = None
_client_stack: contextlib.AsyncExitStack | None = None
_client_loop: asyncio.AbstractEventLoop | None = None
_client_lock: asyncio.Lock | None = None


def _new_client_cm(config: Config) -> Any:
    return aioboto3.Session().client(  # pyright: ignore[reportGeneralTypeIssues]
        "s3",
        endpoint_url=get_s3_endpoint_url(),
        region_name="garage",
        aws_access_key_id=get_s3_access_key(),
        aws_secret_access_key=get_s3_secret_key(),
        config=config,
    )


async def _get_client() -> Any:
    """The process's shared S3 client, opened lazily on the running loop.

    A client is tied to the loop it was opened on, so a different loop (a test's, or a loop
    recreated after close) gets a fresh one; the stale client is dropped, not closed, since
    closing it needs its own loop.
    """
    global _client, _client_stack, _client_loop, _client_lock
    loop = asyncio.get_running_loop()
    if _client_loop is not loop:
        _client = _client_stack = None
        _client_loop = loop
        _client_lock = asyncio.Lock()
    if _client is not None:
        return _client
    assert _client_lock is not None
    async with _client_lock:
        if _client is None:
            stack = contextlib.AsyncExitStack()
            _client = await stack.enter_async_context(_new_client_cm(_WORKER_CLIENT_CONFIG))
            _client_stack = stack
    return _client


def reset_client() -> None:
    """Forget any inherited client (call from worker_process_init, after fork)."""
    global _client, _client_stack, _client_loop, _client_lock
    _client = _client_stack = _client_loop = _client_lock = None


async def close_client() -> None:
    """Close the shared client (call from worker_process_shutdown, on the worker loop)."""
    global _client, _client_stack
    stack = _client_stack
    _client = _client_stack = None
    if stack is not None and _client_loop is asyncio.get_running_loop():
        await stack.aclose()


def _sanitize_filename(filename: str) -> str:
    """Keep alphanumeric, dots, hyphens, underscores; fallback to 'document.pdf'."""
    safe = re.sub(r"[^\w.\-]", "_", filename).strip()
    return safe if safe else "document.pdf"


def build_raw_storage_key(user_id: UUID, doc_id: UUID, filename: str) -> str:
    """Deterministic raw PDF storage key, computable before the upload happens."""
    return f"raw/{user_id}/{doc_id}/{_sanitize_filename(filename)}"


def build_picture_crop_key(document_id: str, self_ref: str) -> str:
    """Deterministic picture-crop storage key, computable before the upload happens.

    `self_ref` is a Docling JSON-pointer like "#/pictures/3"; the leading "#/" is stripped
    so the key reads as "{document_id}/pictures/3.png" rather than carrying a "#" segment.
    """
    return f"{document_id}/{self_ref.lstrip('#/')}.png"


async def upload_pdf(
    user_id: UUID,
    doc_id: UUID,
    filename: str,
    fileobj,
) -> str:
    """
    Upload PDF to S3/Garage. Returns storage_key.
    Key format: raw/{user_id}/{doc_id}/{sanitized_filename}
    """
    storage_key = build_raw_storage_key(user_id, doc_id, filename)
    # Per-call client: this runs in the API process, which the shared worker client is not for.
    async with _new_client_cm(_CLIENT_CONFIG) as client:
        # `fileobj` is Starlette's SpooledTemporaryFile, which spills to disk above 1 MB —
        # so for all but the smallest PDFs these are real blocking disk reads. Handing it
        # to aioboto3 as Body= makes botocore read it from the event loop, freezing every
        # open SSE stream for the length of the upload. Read it off-loop instead.
        # upload_bytes() is not reused here: it takes bytes already in hand, and the read
        # is exactly the part that must not happen on the loop.
        def _read_all() -> bytes:
            with contextlib.suppress(Exception):
                fileobj.seek(0)
            return fileobj.read()

        body = await asyncio.to_thread(_read_all)

        await client.put_object(
            Bucket=get_s3_raw_bucket(),
            Key=storage_key,
            Body=body,
            ContentType="application/pdf",
            # Trust the bytes we actually read over the caller's advisory count: a
            # mismatched ContentLength makes Garage reject or truncate the object.
            ContentLength=len(body),
        )
    return storage_key


_STREAM_CHUNK_SIZE = 1024 * 256  # 256 KB


async def download_file(storage_key: str, *, bucket: str | None = None) -> Path:
    """Download an object to a local tempfile and return its path. Streams to disk to avoid loading entire file into memory."""
    suffix = Path(storage_key).suffix or ".bin"
    target_bucket = bucket or get_s3_raw_bucket()
    client = await _get_client()
    resp = await client.get_object(Bucket=target_bucket, Key=storage_key)
    body = resp["Body"]
    with tempfile.NamedTemporaryFile(prefix="s3_", suffix=suffix, delete=False) as f:
        while True:
            chunk = body.read(_STREAM_CHUNK_SIZE)
            if hasattr(chunk, "__await__"):
                chunk = await chunk
            if not chunk:
                break
            f.write(chunk)
        return Path(f.name)


async def upload_file(
    path: Path,
    key: str,
    content_type: str,
    *,
    bucket: str | None = None,
) -> str:
    """Upload a local file to S3/Garage by path. Returns key.

    Streams from disk (aioboto3 reads it through aiofiles, multipart above the transfer
    threshold), so the artifact is never held in memory whole.
    """
    client = await _get_client()
    await client.upload_file(
        str(path),
        bucket or get_s3_bucket(),
        key,
        ExtraArgs={"ContentType": content_type},
    )
    return key


async def upload_picture_crop(
    document_id: str,
    self_ref: str,
    data: bytes,
    *,
    label: str | None = None,
    confidence: float | None = None,
) -> str:
    """Upload one picture crop PNG to the pictures bucket.

    Tags the object with its Phase-4 classification label/confidence as S3 user metadata
    (not a separate index file) so Phase 11's router can `head_object` to decide how to
    route a picture without downloading and opening the image.
    """
    key = build_picture_crop_key(document_id, self_ref)
    metadata: dict[str, str] = {}
    if label is not None:
        metadata["classification-label"] = label
    if confidence is not None:
        metadata["classification-confidence"] = f"{confidence:.4f}"
    client = await _get_client()
    await client.put_object(
        Bucket=get_s3_pictures_bucket(),
        Key=key,
        Body=data,
        ContentType="image/png",
        ContentLength=len(data),
        Metadata=metadata,
    )
    return key
