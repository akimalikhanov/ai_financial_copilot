"""Ingestion pipeline task."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any
from uuid import UUID

from celery.exceptions import Retry, SoftTimeLimitExceeded
from celery.signals import setup_logging, worker_process_init, worker_process_shutdown
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from src.api.logging import configure_worker_logging
from src.celery_app import celery_app
from src.observability.metrics import (
    INGESTION_CHUNKS,
    INGESTION_DOCUMENTS,
    INGESTION_DURATION,
    INGESTION_QUEUE_WAIT,
)
from src.redis_client import ingestion_stream_key
from src.services.ingestion.chunker import reset_tokenizer
from src.services.ingestion.docling_parser import reset_converter
from src.services.ingestion.embedder import reset_clients as reset_embedding_clients
from src.services.ingestion.opensearch_ingest import reset_client as reset_opensearch_client
from src.services.ingestion.picture_enricher import enrich_pictures as _enrich_pictures
from src.services.ingestion.picture_enricher import reset as reset_picture_enricher
from src.services.ingestion.picture_enricher import (
    validate_config as validate_picture_enricher_config,
)
from src.services.ingestion.qdrant_ingest import reset_client as reset_qdrant_client
from src.services.ingestion.s3_client import close_client as close_s3_client
from src.services.ingestion.s3_client import reset_client as reset_s3_client
from src.services.ingestion.table_summarizer import reset as reset_table_summarizer
from src.services.ingestion.table_summarizer import (
    summarize_table_chunks as _summarize_table_chunks,
)
from src.services.llm_router import get_router
from src.services.prompts.prompt_loader import get_prompt_loader
from src.utils.config import (
    get_db_url,
    get_docling_parse_timeout,
    get_embedding_dim,
    get_embedding_model,
    get_ingest_max_pages,
    get_picture_enricher_enabled,
    get_redis_app_url,
    get_s3_chunks_bucket,
    get_s3_docling_bucket,
    get_s3_rendered_bucket,
    get_table_summarizer_enabled,
)

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument

logger = logging.getLogger(__name__)
_worker_loop: asyncio.AbstractEventLoop | None = None
_redis_ingestion: Redis | None = None
_engine = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _get_int_env(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    return int(raw)


# Ingestion gets its own limits rather than the app-wide CELERY_TASK_*_SECONDS pair, which is
# sized for chat (360/450). A 1000-page parse alone outlives that, so the global limits made
# every inner stage timeout unreachable: Celery reaped the child first and every large-document
# failure surfaced as a generic SoftTimeLimitExceeded instead of the stage that actually blew.
# Must stay strictly above the sum of the stage budgets — see docs/stages/
# ingestion-pipeline-findings.md P1-2 for the ladder.
_task_soft_time_limit = _get_int_env("INGEST_TASK_SOFT_TIME_LIMIT_SECONDS") or 2400
# Held back from the enrichment budget for everything after it (~60s on a 1043-page filing).
_POST_ENRICH_RESERVE_SECONDS = 300
_task_time_limit = _get_int_env("INGEST_TASK_TIME_LIMIT_SECONDS") or 2700
INGEST_MAX_ATTEMPTS = int(os.getenv("INGEST_MAX_ATTEMPTS", "3"))


@setup_logging.connect
def _on_celery_setup_logging(**_kwargs: object) -> None:
    configure_worker_logging()


@worker_process_init.connect
def _on_worker_process_init(**_kwargs: object) -> None:
    global _worker_loop, _redis_ingestion, _engine, _session_factory
    configure_worker_logging()
    reset_converter()
    reset_tokenizer()
    reset_embedding_clients()
    reset_table_summarizer()
    reset_picture_enricher()
    get_router.cache_clear()
    get_prompt_loader.cache_clear()
    reset_qdrant_client()
    reset_opensearch_client()
    reset_s3_client()
    if get_picture_enricher_enabled():
        validate_picture_enricher_config()
    if _worker_loop is None or _worker_loop.is_closed():
        _worker_loop = asyncio.new_event_loop()
    if _redis_ingestion is None:
        _redis_ingestion = Redis.from_url(get_redis_app_url(), decode_responses=True)
    if _engine is None:
        _engine = create_async_engine(get_db_url(), poolclass=NullPool)
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            _engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
        )


@worker_process_shutdown.connect
def _on_worker_process_shutdown(**_kwargs: object) -> None:
    global _worker_loop, _redis_ingestion, _engine, _session_factory
    if _worker_loop is None or _worker_loop.is_closed():
        return
    if _redis_ingestion is not None:
        _worker_loop.run_until_complete(_redis_ingestion.aclose())
    if _engine is not None:
        _worker_loop.run_until_complete(_engine.dispose())
    _worker_loop.run_until_complete(close_s3_client())
    _redis_ingestion = None
    _engine = None
    _session_factory = None
    _worker_loop.close()
    _worker_loop = None


def _enforce_page_limit(pdf_path: Path) -> None:
    """Fail a document that is too large to parse, before any model runs.

    Cheap precheck, because the expensive failure is not a slow parse: a document past the
    memory ceiling OOM-kills the container, and task_acks_late redelivers it until every
    INGEST_MAX_ATTEMPTS is spent on container restarts, with no record of why. A PDF whose page
    count cannot be read is let through — the parse is the better judge of a broken file.
    """
    from src.services.ingestion import docling_parser

    max_pages = get_ingest_max_pages()
    if max_pages <= 0:
        return
    pages = docling_parser.probe_page_count(pdf_path)
    if pages is not None and pages > max_pages:
        raise RuntimeError(
            f"Document has {pages} pages, above the {max_pages}-page ingestion limit "
            f"(INGEST_MAX_PAGES)"
        )


def _export_artifacts(document, out_dir: Path) -> tuple[Path, Path]:
    """Write the DoclingDocument to `out_dir` as JSON and Markdown, returning both paths.

    Clears every `pic.image` first, so it must run after `_upload_picture_crops`.
    ImageRefMode.PLACEHOLDER only strips images from Markdown: `save_as_json` still serializes
    each crop as a base64 data URI (~1.33x its bytes, a second copy of the pictures bucket).
    Nothing downstream reads the crops — enrichment has run and the chunker serializes
    pictures as placeholders — so dropping them also frees their memory before chunking.
    `indent=None` because save_as_json builds the whole JSON string before writing it.
    """
    from docling_core.types.doc.base import ImageRefMode

    for pic in document.pictures:
        pic.image = None
    json_p = out_dir / "docling.json"
    md_p = out_dir / "document.md"
    document.save_as_json(json_p, image_mode=ImageRefMode.PLACEHOLDER, indent=None)  # pyright: ignore[reportArgumentType]
    document.save_as_markdown(md_p, image_mode=ImageRefMode.PLACEHOLDER)
    return json_p, md_p


def _encode_picture_crop(pic) -> dict[str, Any] | None:
    """Encode one picture's in-memory crop to PNG bytes, with its top classification label if
    Docling produced one (do_picture_classification, see docling_parser.py). None when the
    picture has no crop or it fails to encode.

    Crops only exist here because generate_picture_images=True keeps them on the
    DoclingDocument through parse, until `_export_artifacts` clears them.
    """
    if pic.image is None:
        return None
    try:
        pil_image = pic.image.pil_image
        if pil_image is None:
            raise ValueError("pil_image decode returned None")
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        label: str | None = None
        confidence: float | None = None
        if pic.meta is not None and pic.meta.classification is not None:
            main = pic.meta.classification.get_main_prediction()
            label, confidence = main.class_name, main.confidence
        return {
            "self_ref": pic.self_ref,
            "data": buf.getvalue(),
            "label": label,
            "confidence": confidence,
        }
    except Exception:
        # A crop is a re-runnable convenience artifact (Phase 5), not required for
        # this document to become searchable — never fail the pipeline over one.
        logger.warning(
            "pipeline.picture_crop_encode_failed",
            extra={"self_ref": pic.self_ref},
            exc_info=True,
        )
        return None


# Crops encoded and in flight at once. Bounds both the PNG bytes held and the concurrent
# requests against Garage; unbounded, a 300-picture document opened 300 uploads together.
_CROP_UPLOAD_CONCURRENCY = 8


async def _load_persisted_document(key: str, document_id: str) -> DoclingDocument | None:
    """The docling.json an earlier attempt uploaded, or None when it is missing or unreadable
    (the caller then re-parses)."""
    from docling_core.types.doc.document import DoclingDocument as _DoclingDocument

    from src.services.ingestion import s3_client

    try:
        path = await s3_client.download_file(key, bucket=get_s3_docling_bucket())
    except Exception:
        logger.info("pipeline.resume_unavailable", extra={"document_id": document_id})
        return None
    try:
        document = await asyncio.to_thread(_DoclingDocument.load_from_json, path)
    except Exception:
        logger.warning(
            "pipeline.resume_load_failed", extra={"document_id": document_id}, exc_info=True
        )
        return None
    finally:
        path.unlink(missing_ok=True)
    logger.info(
        "pipeline.resumed_from_artifact",
        extra={"document_id": document_id, "pictures": len(document.pictures)},
    )
    return document


async def _upload_picture_crops(document, document_id: str) -> int:
    """Encode, upload and release each picture crop, returning how many landed in S3.

    Each crop is encoded under the semaphore and dropped as soon as its upload returns, so at
    most `_CROP_UPLOAD_CONCURRENCY` encoded crops are live — never a list of all of them.
    """
    from src.services.ingestion import s3_client

    sem = asyncio.Semaphore(_CROP_UPLOAD_CONCURRENCY)

    async def _one(pic) -> bool:
        async with sem:
            crop = await asyncio.to_thread(_encode_picture_crop, pic)
            if crop is None:
                return False
            # A crop is a re-runnable convenience artifact (Phase 5), not required for
            # this document to become searchable — never fail the pipeline over one.
            try:
                await s3_client.upload_picture_crop(
                    document_id,
                    crop["self_ref"],
                    crop["data"],
                    label=crop["label"],
                    confidence=crop["confidence"],
                )
            except Exception:
                logger.warning(
                    "pipeline.picture_crop_upload_failed",
                    extra={"document_id": document_id, "self_ref": crop["self_ref"]},
                    exc_info=True,
                )
                return False
            return True

    results = await asyncio.gather(*(_one(pic) for pic in document.pictures))
    return sum(results)


def _write_chunks_jsonl(path: Path, chunks: list[dict[str, Any]], db_chunks: list[Any]) -> None:
    """Stream the chunk backup to disk one row at a time, rather than joining it into one
    string and encoding a second copy."""
    with path.open("w", encoding="utf-8") as f:
        for i, (c, db) in enumerate(zip(chunks, db_chunks, strict=True)):
            if i:
                f.write("\n")
            f.write(
                json.dumps(
                    {
                        "chunk_id": str(db.id),
                        "chunk_index": c["chunk_index"],
                        "raw_text": c["raw_text"],
                        "enriched_text": c["enriched_text"],
                        "heading_trail": c.get("heading_trail"),
                        "chunk_type": c.get("chunk_type"),
                        "page_start": c.get("page_start"),
                        "page_end": c.get("page_end"),
                        "token_count": c.get("token_count"),
                        "table_nl_summary": c.get("table_nl_summary"),
                        "table_nl_summary_model": c.get("table_nl_summary_model"),
                        "provenance": c.get("provenance"),
                        "metadata": c.get("metadata", {}),
                    },
                    ensure_ascii=False,
                    default=str,
                )
            )


async def _run_pipeline(document_id: str) -> None:  # noqa: C901
    from src.repository.chunk_repository import ChunkRepository
    from src.repository.document_repository import DocumentRepository
    from src.services.ingestion import (
        chunker,
        docling_parser,
        embedder,
        opensearch_ingest,
        qdrant_ingest,
        s3_client,
    )

    if _session_factory is None:
        raise RuntimeError("Ingestion worker DB session factory is not initialized")
    sf = _session_factory
    doc_uuid = UUID(document_id)
    pdf_path: Path | None = None
    work_dir: Path | None = None
    pipeline_started_at = perf_counter()
    stage_start = perf_counter()
    stage_times: dict[str, float] = {}
    stage_order: list[str] = []
    stage_total = 13 + get_table_summarizer_enabled() + get_picture_enricher_enabled()
    stage_index = 0
    current_stage = "initializing"
    upload_metadata: dict = {}
    stream_key = ingestion_stream_key(document_id)

    async def _emit(event_type: str, data: dict) -> None:
        if _redis_ingestion is None:
            return
        try:
            payload = json.dumps({"type": event_type, **data})
            await _redis_ingestion.xadd(stream_key, {"payload": payload}, "*", maxlen=100)
        except Exception:
            pass

    async def _log_stage(stage_name: str) -> None:
        nonlocal stage_index, current_stage, stage_start
        if current_stage != "initializing":
            stage_times[current_stage] = round(perf_counter() - stage_start, 3)
        stage_index += 1
        current_stage = stage_name
        stage_order.append(stage_name)
        stage_start = perf_counter()
        logger.info(
            f"pipeline.stage [{stage_index}/{stage_total}] {stage_name}",
            extra={
                "document_id": document_id,
                "stage": stage_name,
                "stage_index": stage_index,
                "stage_total": stage_total,
            },
        )
        await _emit(
            "stage", {"stage": stage_name, "stage_index": stage_index, "stage_total": stage_total}
        )

    def _flush_stage_times() -> None:
        if current_stage != "initializing":
            stage_times[current_stage] = round(perf_counter() - stage_start, 3)

    def _format_stage_times() -> dict[str, float]:
        """Return stages dict numbered in execution order, sorted by stage number."""
        formatted: dict[str, float] = {}
        for idx, name in enumerate(stage_order, start=1):
            if name in stage_times:
                formatted[f"{idx:02d}. {name}"] = stage_times[name]
        return dict(sorted(formatted.items()))

    async def _timed(stage: str, coro):
        """Await coro, recording its latency under INGESTION_DURATION[stage]."""
        t = perf_counter()
        try:
            return await coro
        finally:
            INGESTION_DURATION.labels(stage).observe(perf_counter() - t)

    try:
        # -- fetch document record, set status -> processing ----------------
        await _log_stage("fetch_document_record")
        async with sf() as session:
            repo = DocumentRepository(session)
            doc = await repo.get_by_id(doc_uuid)
            if doc is None:
                raise LookupError(f"Document {document_id} not found")

            attempt = await repo.increment_attempt_count(doc_uuid)
            if attempt > INGEST_MAX_ATTEMPTS:
                await repo.set_failed(
                    doc_uuid,
                    f"Exceeded max ingestion attempts ({INGEST_MAX_ATTEMPTS})",
                )
                await session.commit()
                INGESTION_DOCUMENTS.labels("failed").inc()
                logger.warning(
                    "pipeline.max_attempts_exceeded",
                    extra={
                        "document_id": document_id,
                        "attempt": attempt,
                        "max_attempts": INGEST_MAX_ATTEMPTS,
                    },
                )
                return

            if attempt == 1:
                # First attempt only: on a redelivery created_at is the original upload,
                # so the difference would report the failed attempt's runtime as queue wait.
                INGESTION_QUEUE_WAIT.observe((datetime.now(UTC) - doc.created_at).total_seconds())

            storage_key = doc.storage_key
            user_id = str(doc.user_id)
            prior_parse_status = doc.parse_status
            upload_metadata = dict(doc.document_metadata or {})
            await repo.update_status(doc_uuid, "processing", clear_processing_error=True)
            await session.commit()

        work_dir = Path(tempfile.mkdtemp(prefix="ingest_"))
        base_key = f"processed/{user_id}/{document_id}"

        # -- resume: a retry reuses the parse an earlier attempt persisted ----
        # parse_status is written in the same step that uploads docling.json, which already
        # carries the picture descriptions, and the crops are uploaded before it. So a retry
        # that finds both skips download, parse, enrichment and export entirely.
        document = None
        if attempt > 1 and prior_parse_status is not None:
            await _log_stage("load_persisted_parse")
            document = await _load_persisted_document(f"{base_key}/docling.json", document_id)
        if document is not None:
            parse_status = prior_parse_status
            # load_persisted_parse stands in for the parse path's own stages.
            stage_total -= 4 + get_picture_enricher_enabled()
        else:
            # -- download raw PDF -----------------------------------------------
            await _log_stage("download_pdf")
            pdf_path = await s3_client.download_file(storage_key)

            # -- page-count guardrail (milliseconds, no models) ------------------
            await asyncio.to_thread(_enforce_page_limit, pdf_path)

            # -- parse with Docling (CPU/GPU-bound) -----------------------------
            await _log_stage("parse_pdf_docling")
            # Docling's own document_timeout bounds its page loop only; assembly, reading order and
            # enrichment run outside it. This is the wall-clock ceiling on the whole parse. A thread
            # cannot be killed, so the abandoned parse runs on until it finishes or Celery's hard
            # time limit reaps the child; the document fails now rather than hanging.
            parse_timeout = get_docling_parse_timeout()
            try:
                parse_result = await _timed(
                    "parse",
                    asyncio.wait_for(
                        asyncio.to_thread(docling_parser.parse, pdf_path), timeout=parse_timeout
                    ),
                )
            except TimeoutError as exc:
                raise RuntimeError(
                    f"Docling parse exceeded the {parse_timeout}s wall-clock limit "
                    f"(DOCLING_PARSE_TIMEOUT_SECONDS)"
                ) from exc

            # -- describe pictures with a vision model (network-bound) ---------
            # Must run before export and before chunking: it writes pic.meta.description, which
            # the exported JSON carries and the chunker substitutes for `<!-- image -->`.
            if get_picture_enricher_enabled():
                await _log_stage("enrich_pictures")
                try:
                    # The enricher sizes its own budget from the picture count and keeps completed
                    # batches on timeout; this cap keeps it inside the task's soft limit.
                    remaining = (
                        _task_soft_time_limit
                        - (perf_counter() - pipeline_started_at)
                        - _POST_ENRICH_RESERVE_SECONDS
                    )
                    described = await _enrich_pictures(parse_result.document, max_timeout=remaining)
                    logger.info(
                        "pipeline.pictures_enriched",
                        extra={"document_id": document_id, "described": described},
                    )
                except Exception:
                    # Enrichment degrades to Phase 4 quality; it never fails a document.
                    logger.warning(
                        "pipeline.picture_enrichment_failed",
                        extra={"document_id": document_id},
                        exc_info=True,
                    )

            # -- upload picture crops (bounded, one encoded crop per slot) -------
            # Must run before export: export clears pic.image so docling.json carries no base64.
            await _log_stage("upload_picture_crops")
            crops_uploaded = await _upload_picture_crops(parse_result.document, document_id)
            logger.info(
                "pipeline.picture_crops_uploaded",
                extra={
                    "document_id": document_id,
                    "uploaded": crops_uploaded,
                    "pictures": len(parse_result.document.pictures),
                },
            )

            # -- export artifacts to disk (CPU-bound serialization) -------------
            await _log_stage("export_docling_artifacts")
            json_path, md_path = await asyncio.to_thread(
                _export_artifacts, parse_result.document, work_dir
            )

            # -- update metadata + upload artifacts (parallel I/O) --------------
            await _log_stage("save_metadata_and_upload_artifacts")

            async def _save_metadata():
                async with sf() as session:
                    repo = DocumentRepository(session)
                    merged_metadata = {
                        **upload_metadata,
                        **parse_result.metadata,
                    }
                    await repo.update_metadata(
                        doc_uuid,
                        page_count=parse_result.page_count,
                        extracted_title=parse_result.extracted_title,
                        parse_status=parse_result.parse_status,
                        metadata=merged_metadata,
                    )
                    await session.commit()

            await asyncio.gather(
                _save_metadata(),
                s3_client.upload_file(
                    json_path,
                    f"{base_key}/docling.json",
                    "application/json",
                    bucket=get_s3_docling_bucket(),
                ),
                s3_client.upload_file(
                    md_path,
                    f"{base_key}/document.md",
                    "text/markdown",
                    bucket=get_s3_rendered_bucket(),
                ),
            )
            json_path.unlink(missing_ok=True)
            md_path.unlink(missing_ok=True)
            document = parse_result.document
            parse_status = parse_result.parse_status

        # -- chunk document (CPU-bound) -------------------------------------
        await _log_stage("chunk_document")
        chunks = await _timed(
            "chunk", asyncio.to_thread(chunker.chunk_document, document, document_id)
        )
        INGESTION_CHUNKS.observe(len(chunks))

        if not chunks:
            # Zero chunks means nothing was indexed: the document is "ready" but no query can
            # ever retrieve it. Logged at warning and counted under its own status rather than
            # "success" — a scanned PDF with OCR disabled lands here, and as a plain success it
            # was indistinguishable from a document that ingested correctly.
            logger.warning(
                "pipeline.no_chunks",
                extra={
                    "document_id": document_id,
                    "stage": "chunk_document",
                    "parse_status": parse_status,
                },
            )
            await _log_stage("finalize_ready")
            stage_times["finalize_ready"] = round(perf_counter() - stage_start, 3)
            ingest_times = {
                "stages": _format_stage_times(),
                "total_time": round(perf_counter() - pipeline_started_at, 3),
            }
            async with sf() as session:
                repo = DocumentRepository(session)
                await repo.update_status(doc_uuid, "ready")
                await repo.set_ingest_time_seconds(doc_uuid, ingest_times)
                await session.commit()
            INGESTION_DOCUMENTS.labels("no_content").inc()
            await _emit("done", {"chunks": 0})
            logger.info(
                "pipeline.complete",
                extra={
                    "document_id": document_id,
                    "chunks": 0,
                    "ingest_times": ingest_times,
                },
            )
            return

        # -- summarize table chunks (LLM call, optional) ----------------------
        if get_table_summarizer_enabled():
            await _log_stage("summarize_table_chunks")
            chunks = await _summarize_table_chunks(chunks)

        # -- persist chunks to Postgres -------------------------------------
        await _log_stage("persist_chunks_postgres")
        embedding_model = get_embedding_model()
        for c in chunks:
            c["embedding_model"] = embedding_model

        async with sf() as session:
            if await DocumentRepository(session).get_by_id(doc_uuid) is None:
                raise LookupError(f"Document {document_id} no longer exists")
            chunk_repo = ChunkRepository(session)
            db_chunks = await chunk_repo.create_many(doc_uuid, chunks)
            await session.commit()

        # -- generate embeddings (CPU/GPU-bound) ----------------------------
        # When summarizer is enabled, table chunks embed their NL summary;
        # otherwise (or on summarization failure) fall back to enriched_text.
        await _log_stage("embed_chunks")
        texts = [c.get("table_nl_summary") or c["enriched_text"] for c in chunks]
        vectors = await _timed("embed", asyncio.to_thread(embedder.embed_chunks, texts))

        # -- prepare Qdrant payload -----------------------------------------
        chunks_with_vectors = [
            {
                "vector": vec,
                "chunk_id": db_chunk.id,
                "chunk_index": c["chunk_index"],
                "chunk_type": c.get("chunk_type"),
                "page_start": c.get("page_start"),
                "page_end": c.get("page_end"),
                "heading_trail": c.get("heading_trail"),
            }
            for c, vec, db_chunk in zip(chunks, vectors, db_chunks, strict=True)
        ]

        # -- prepare OpenSearch payload -------------------------------------
        os_chunks = [
            {
                "chunk_id": db_chunk.id,
                "chunk_index": c["chunk_index"],
                "enriched_text": c["enriched_text"],
                "heading_trail": c.get("heading_trail"),
                "chunk_type": c.get("chunk_type"),
                "page_start": c.get("page_start"),
                "page_end": c.get("page_end"),
                "metadata": c.get("metadata", {}),
            }
            for c, db_chunk in zip(chunks, db_chunks, strict=True)
        ]

        # -- ensure collections/indices exist (parallel) --------------------
        await _log_stage("ensure_vector_and_search_indexes")
        dim = len(vectors[0]) if len(vectors) else (get_embedding_dim() or 384)
        await asyncio.gather(
            asyncio.to_thread(qdrant_ingest.ensure_collection, "documents", dim),
            asyncio.to_thread(opensearch_ingest.ensure_index, "chunks"),
        )

        # -- purge any prior generation of this document's chunks ------------
        # By document_id and *before* indexing, so re-ingestion is idempotent wherever a
        # prior attempt died. The old chunk-id list came from Postgres and ran after
        # indexing, so a rollback erased the only record of what needed cleaning.
        # Skipped on the first attempt: ingest_attempt_count is never reset, so nothing can
        # have been indexed under this document_id yet.
        await _log_stage("purge_stale_chunks")
        if attempt > 1:
            await asyncio.gather(
                asyncio.to_thread(qdrant_ingest.delete_by_document, "documents", document_id),
                asyncio.to_thread(opensearch_ingest.delete_by_document, "chunks", document_id),
            )

        # -- index + backup (Qdrant, OpenSearch, S3 chunks.jsonl — parallel)
        await _log_stage("index_and_backup_chunks")
        chunks_jsonl_path = work_dir / "chunks.jsonl"

        async def _backup_chunks() -> None:
            await asyncio.to_thread(_write_chunks_jsonl, chunks_jsonl_path, chunks, db_chunks)
            await s3_client.upload_file(
                chunks_jsonl_path,
                f"{base_key}/chunks.jsonl",
                "application/jsonl",
                bucket=get_s3_chunks_bucket(),
            )

        await asyncio.gather(
            _timed(
                "upsert_qdrant",
                asyncio.to_thread(
                    qdrant_ingest.upsert_chunks,
                    "documents",
                    document_id,
                    chunks_with_vectors,
                    user_id=user_id,
                ),
            ),
            _timed(
                "upsert_opensearch",
                asyncio.to_thread(
                    opensearch_ingest.bulk_index,
                    "chunks",
                    document_id,
                    os_chunks,
                    user_id=user_id,
                ),
            ),
            _backup_chunks(),
        )

        # -- finalize -> ready ----------------------------------------------
        await _log_stage("finalize_ready")
        stage_times["finalize_ready"] = round(perf_counter() - stage_start, 3)
        ingest_times = {
            "stages": _format_stage_times(),
            "total_time": round(perf_counter() - pipeline_started_at, 3),
        }
        async with sf() as session:
            repo = DocumentRepository(session)
            await repo.update_status(doc_uuid, "ready")
            await repo.set_ingest_time_seconds(doc_uuid, ingest_times)
            await session.commit()

        INGESTION_DOCUMENTS.labels("success").inc()
        await _emit("done", {"chunks": len(chunks)})
        logger.info(
            "pipeline.complete",
            extra={
                "document_id": document_id,
                "chunks": len(chunks),
                "ingest_times": ingest_times,
            },
        )

    except LookupError:
        raise
    except SoftTimeLimitExceeded:
        INGESTION_DOCUMENTS.labels("failed").inc()
        _flush_stage_times()
        elapsed = round(perf_counter() - pipeline_started_at, 1)
        msg = (
            f"Ingestion timed out after {elapsed}s at stage '{current_stage}' "
            f"(soft_time_limit={_task_soft_time_limit}s)"
        )
        ingest_times = {
            "stages": _format_stage_times(),
            "total_time": round(perf_counter() - pipeline_started_at, 3),
            "failed_at_stage": current_stage,
        }
        logger.error(
            "pipeline.soft_time_limit",
            extra={
                "document_id": document_id,
                "stage": current_stage,
                "elapsed_seconds": elapsed,
                "stage_times": _format_stage_times(),
            },
        )
        try:
            async with sf() as session:
                repo = DocumentRepository(session)
                await repo.set_failed(doc_uuid, msg)
                await repo.set_ingest_time_seconds(doc_uuid, ingest_times)
                await session.commit()
            await _emit("error", {"message": msg})
        except Exception:
            logger.exception("pipeline.set_failed_error", extra={"document_id": document_id})
        raise
    except Exception as exc:
        INGESTION_DOCUMENTS.labels("failed").inc()
        _flush_stage_times()
        ingest_times = {
            "stages": _format_stage_times(),
            "total_time": round(perf_counter() - pipeline_started_at, 3),
            "failed_at_stage": current_stage,
        }
        logger.exception(
            "pipeline.failed_at_stage",
            extra={
                "document_id": document_id,
                "stage": current_stage,
                "stage_times": _format_stage_times(),
            },
        )
        try:
            async with sf() as session:
                repo = DocumentRepository(session)
                await repo.set_failed(doc_uuid, str(exc))
                await repo.set_ingest_time_seconds(doc_uuid, ingest_times)
                await session.commit()
            await _emit("error", {"message": "Document processing failed. Please try again."})
        except Exception:
            logger.exception("pipeline.set_failed_error", extra={"document_id": document_id})
        raise
    finally:
        if pdf_path is not None:
            pdf_path.unlink(missing_ok=True)
        if work_dir is not None:
            shutil.rmtree(work_dir, ignore_errors=True)


@celery_app.task(
    bind=True,
    name="ingest_document",
    soft_time_limit=_task_soft_time_limit,
    time_limit=_task_time_limit,
)
def ingest_document(self, document_id: str) -> None:
    """Full ingestion pipeline: parse -> chunk -> embed -> index -> finalize."""
    logger.info("ingest_document.start", extra={"document_id": document_id})
    try:
        if _worker_loop is None or _worker_loop.is_closed():
            raise RuntimeError("Ingestion worker loop is not initialized")
        _worker_loop.run_until_complete(_run_pipeline(document_id))
        logger.info("ingest_document.done", extra={"document_id": document_id})
    except LookupError:
        logger.warning(
            "ingest_document.not_found_retrying",
            extra={
                "document_id": document_id,
                "attempt": getattr(self.request, "retries", 0),
            },
        )
        raise self.retry(countdown=1, max_retries=10) from None
    except Retry:
        raise
    except SoftTimeLimitExceeded:
        logger.error(
            "ingest_document.soft_time_limit",
            extra={"document_id": document_id},
        )
        raise
    except Exception:
        logger.exception("ingest_document.failed", extra={"document_id": document_id})
        raise
