"""S3/Garage client for PDF uploads."""

from __future__ import annotations

import re
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
    body: bytes,
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
        await client.put_object(
            Bucket=get_s3_bucket(),
            Key=storage_key,
            Body=body,
            ContentType="application/pdf",
        )
    return storage_key
