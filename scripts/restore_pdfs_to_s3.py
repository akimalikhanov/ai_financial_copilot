"""Re-upload PDFs from data/pdfs/ to Garage S3 using the storage_key already in the DB."""

import asyncio
import os
from pathlib import Path

import asyncpg
import boto3
from botocore.config import Config

PDF_DIR = Path(__file__).parent.parent / "data" / "pdfs"

DB_URL = (
    f"postgresql://{os.environ['APP_DB_USER']}:{os.environ['APP_DB_PASSWORD']}"
    f"@{os.environ.get('POSTGRES_HOST', '127.0.0.1')}:{os.environ.get('POSTGRES_PORT', '5434')}"
    f"/{os.environ['APP_DB']}"
)

S3 = boto3.client(
    "s3",
    endpoint_url=os.environ["AWS_ENDPOINT_URL"],
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    region_name=os.environ.get("AWS_DEFAULT_REGION", "garage"),
    config=Config(signature_version="s3v4"),
)
BUCKET = os.environ.get("S3_RAW_BUCKET", "pdfs")


async def main() -> None:
    conn = await asyncpg.connect(DB_URL)
    rows = await conn.fetch(
        "SELECT original_filename, storage_key FROM documents ORDER BY original_filename"
    )
    await conn.close()

    print(f"Found {len(rows)} documents in DB, bucket={BUCKET}")
    ok = skipped = missing = 0

    for row in rows:
        filename: str = row["original_filename"]
        key: str = row["storage_key"]
        local = PDF_DIR / filename

        if not local.exists():
            print(f"  MISSING locally: {filename}")
            missing += 1
            continue

        # Check if already uploaded
        try:
            S3.head_object(Bucket=BUCKET, Key=key)
            print(f"  already exists: {key}")
            skipped += 1
            continue
        except S3.exceptions.ClientError:
            pass
        except Exception:
            pass

        print(f"  uploading: {filename} → {key}")
        S3.upload_file(
            str(local),
            BUCKET,
            key,
            ExtraArgs={"ContentType": "application/pdf"},
        )
        ok += 1

    print(f"\nDone: {ok} uploaded, {skipped} already existed, {missing} missing locally")


if __name__ == "__main__":
    asyncio.run(main())
