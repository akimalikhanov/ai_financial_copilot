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
    get_s3_secret_key,
)


def _sanitize_filename(filename: str) -> str:
    """Keep alphanumeric, dots, hyphens, underscores; fallback to 'document.pdf'."""
    safe = re.sub(r"[^\w.\-]", "_", filename).strip()
    return safe if safe else "document.pdf"


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
    storage_key = f"raw/{user_id}/{doc_id}/{_sanitize_filename(filename)}"
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
            Bucket=get_s3_bucket(),
            Key=storage_key,
            Body=fileobj,
            ContentType="application/pdf",
            **({"ContentLength": content_length} if content_length is not None else {}),
        )
    return storage_key


async def download_file(storage_key: str) -> Path:
    """Download an object to a local tempfile and return its path."""
    suffix = Path(storage_key).suffix or ".bin"
    session = aioboto3.Session()
    async with session.client(  # pyright: ignore[reportGeneralTypeIssues]
        "s3",
        endpoint_url=get_s3_endpoint_url(),
        region_name="garage",
        aws_access_key_id=get_s3_access_key(),
        aws_secret_access_key=get_s3_secret_key(),
    ) as client:
        resp = await client.get_object(Bucket=get_s3_bucket(), Key=storage_key)
        body = resp["Body"]
        data = body.read()
        if hasattr(data, "__await__"):
            data = await data
        with tempfile.NamedTemporaryFile(prefix="s3_", suffix=suffix, delete=False) as f:
            f.write(data)
            return Path(f.name)


async def upload_bytes(key: str, data: bytes, content_type: str) -> str:
    """Upload in-memory bytes to S3/Garage. Returns key."""
    session = aioboto3.Session()
    async with session.client(  # pyright: ignore[reportGeneralTypeIssues]
        "s3",
        endpoint_url=get_s3_endpoint_url(),
        region_name="garage",
        aws_access_key_id=get_s3_access_key(),
        aws_secret_access_key=get_s3_secret_key(),
    ) as client:
        await client.put_object(
            Bucket=get_s3_bucket(),
            Key=key,
            Body=data,
            ContentType=content_type,
            ContentLength=len(data),
        )
    return key
