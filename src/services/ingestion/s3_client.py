"""S3/Garage client for PDF uploads."""

from __future__ import annotations

import contextlib
import re
import tempfile
from pathlib import Path
from uuid import UUID

import aioboto3

from src.utils.config import (
    get_s3_access_key,
    get_s3_bucket,
    get_s3_endpoint_url,
    get_s3_raw_bucket,
    get_s3_secret_key,
)


def _sanitize_filename(filename: str) -> str:
    """Keep alphanumeric, dots, hyphens, underscores; fallback to 'document.pdf'."""
    safe = re.sub(r"[^\w.\-]", "_", filename).strip()
    return safe if safe else "document.pdf"


def build_raw_storage_key(user_id: UUID, doc_id: UUID, filename: str) -> str:
    """Deterministic raw PDF storage key, computable before the upload happens."""
    return f"raw/{user_id}/{doc_id}/{_sanitize_filename(filename)}"


async def upload_pdf(
    user_id: UUID,
    doc_id: UUID,
    filename: str,
    fileobj,
    *,
    content_length: int | None = None,
) -> str:
    """
    Upload PDF to S3/Garage. Returns storage_key.
    Key format: raw/{user_id}/{doc_id}/{sanitized_filename}
    """
    storage_key = build_raw_storage_key(user_id, doc_id, filename)
    session = aioboto3.Session()
    async with session.client(  # pyright: ignore[reportGeneralTypeIssues]
        "s3",
        endpoint_url=get_s3_endpoint_url(),
        region_name="garage",
        aws_access_key_id=get_s3_access_key(),
        aws_secret_access_key=get_s3_secret_key(),
    ) as client:
        with contextlib.suppress(Exception):
            fileobj.seek(0)
        await client.put_object(
            Bucket=get_s3_raw_bucket(),
            Key=storage_key,
            Body=fileobj,
            ContentType="application/pdf",
            **({"ContentLength": content_length} if content_length is not None else {}),
        )
    return storage_key


_STREAM_CHUNK_SIZE = 1024 * 256  # 256 KB


async def download_file(storage_key: str, *, bucket: str | None = None) -> Path:
    """Download an object to a local tempfile and return its path. Streams to disk to avoid loading entire file into memory."""
    suffix = Path(storage_key).suffix or ".bin"
    target_bucket = bucket or get_s3_raw_bucket()
    session = aioboto3.Session()
    async with session.client(  # pyright: ignore[reportGeneralTypeIssues]
        "s3",
        endpoint_url=get_s3_endpoint_url(),
        region_name="garage",
        aws_access_key_id=get_s3_access_key(),
        aws_secret_access_key=get_s3_secret_key(),
    ) as client:
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


async def upload_bytes(
    key: str,
    data: bytes,
    content_type: str,
    *,
    bucket: str | None = None,
) -> str:
    """Upload in-memory bytes to S3/Garage. Returns key."""
    target_bucket = bucket or get_s3_bucket()
    session = aioboto3.Session()
    async with session.client(  # pyright: ignore[reportGeneralTypeIssues]
        "s3",
        endpoint_url=get_s3_endpoint_url(),
        region_name="garage",
        aws_access_key_id=get_s3_access_key(),
        aws_secret_access_key=get_s3_secret_key(),
    ) as client:
        await client.put_object(
            Bucket=target_bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
            ContentLength=len(data),
        )
    return key
