"""Ingestion pipeline task."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from time import perf_counter
from uuid import UUID

from celery.exceptions import Retry, SoftTimeLimitExceeded
from celery.signals import setup_logging, worker_process_init, worker_process_shutdown
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from src.api.logging import configure_worker_logging
from src.celery_app import celery_app
from src.redis_client import ingestion_stream_key
from src.services.ingestion.chunker import reset_tokenizer
from src.services.ingestion.docling_parser import reset_converter
from src.services.ingestion.embedder import reset_clients as reset_embedding_clients
from src.services.ingestion.opensearch_ingest import reset_client as reset_opensearch_client
from src.services.ingestion.qdrant_ingest import reset_client as reset_qdrant_client
from src.services.ingestion.table_summarizer import reset as reset_table_summarizer
from src.services.ingestion.table_summarizer import (
    summarize_table_chunks as _summarize_table_chunks,
)
from src.services.llm_router import get_router
from src.services.prompts.prompt_loader import get_prompt_loader
from src.utils.config import (
    get_db_url,
    get_embedding_dim,
    get_embedding_model,
    get_redis_app_url,
    get_s3_chunks_bucket,
    get_s3_docling_bucket,
    get_s3_rendered_bucket,
    get_table_summarizer_enabled,
)

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


_task_soft_time_limit = _get_int_env("CELERY_TASK_SOFT_TIME_LIMIT_SECONDS")
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
    get_router.cache_clear()
    get_prompt_loader.cache_clear()
    reset_qdrant_client()
    reset_opensearch_client()
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
    _redis_ingestion = None
    _engine = None
    _session_factory = None
    _worker_loop.close()
    _worker_loop = None


def _export_artifacts(document) -> tuple[bytes, bytes]:
    """Serialize DoclingDocument to JSON and Markdown bytes without embedded images."""
    from docling_core.types.doc.base import ImageRefMode

    td = Path(tempfile.mkdtemp(prefix="docling_"))
    try:
        json_p = td / "doc.json"
        md_p = td / "doc.md"
        document.save_as_json(json_p, image_mode=ImageRefMode.PLACEHOLDER)
        document.save_as_markdown(md_p, image_mode=ImageRefMode.PLACEHOLDER)
        return json_p.read_bytes(), md_p.read_bytes()
    finally:
        shutil.rmtree(td, ignore_errors=True)


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
    pipeline_started_at = perf_counter()
    stage_start = perf_counter()
    stage_times: dict[str, float] = {}
    stage_order: list[str] = []
    stage_total = 12 if get_table_summarizer_enabled() else 11
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
                logger.warning(
                    "pipeline.max_attempts_exceeded",
                    extra={
                        "document_id": document_id,
                        "attempt": attempt,
                        "max_attempts": INGEST_MAX_ATTEMPTS,
                    },
                )
                return

            storage_key = doc.storage_key
            user_id = str(doc.user_id)
            upload_metadata = dict(doc.document_metadata or {})
            await repo.update_status(doc_uuid, "processing", clear_processing_error=True)
            await session.commit()

        # -- download raw PDF -----------------------------------------------
        await _log_stage("download_pdf")
        pdf_path = await s3_client.download_file(storage_key)

        # -- parse with Docling (CPU/GPU-bound) -----------------------------
        await _log_stage("parse_pdf_docling")
        parse_result = await asyncio.to_thread(docling_parser.parse, pdf_path)

        # -- export artifacts (CPU-bound serialization) --------------------
        await _log_stage("export_docling_artifacts")
        json_bytes, md_bytes = await asyncio.to_thread(_export_artifacts, parse_result.document)

        # -- update metadata + upload artifacts (parallel I/O) --------------
        await _log_stage("save_metadata_and_upload_artifacts")
        base_key = f"processed/{user_id}/{document_id}"

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
            s3_client.upload_bytes(
                f"{base_key}/docling.json",
                json_bytes,
                "application/json",
                bucket=get_s3_docling_bucket(),
            ),
            s3_client.upload_bytes(
                f"{base_key}/document.md",
                md_bytes,
                "text/markdown",
                bucket=get_s3_rendered_bucket(),
            ),
        )

        # -- chunk document (CPU-bound) -------------------------------------
        await _log_stage("chunk_document")
        chunks = await asyncio.to_thread(chunker.chunk_document, parse_result.document, document_id)

        if not chunks:
            logger.info(
                "pipeline.no_chunks",
                extra={"document_id": document_id, "stage": "chunk_document"},
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
            old_db_chunks = await chunk_repo.list_by_document(doc_uuid)
            old_chunk_ids = [c.id for c in old_db_chunks]
            db_chunks = await chunk_repo.create_many(doc_uuid, chunks)
            await session.commit()

        # -- generate embeddings (CPU/GPU-bound) ----------------------------
        # When summarizer is enabled, table chunks embed their NL summary;
        # otherwise (or on summarization failure) fall back to enriched_text.
        await _log_stage("embed_chunks")
        texts = [c.get("table_nl_summary") or c["enriched_text"] for c in chunks]
        vectors = await asyncio.to_thread(embedder.embed_chunks, texts)

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
        dim = len(vectors[0]) if vectors else (get_embedding_dim() or 384)
        await asyncio.gather(
            asyncio.to_thread(qdrant_ingest.ensure_collection, "documents", dim),
            asyncio.to_thread(opensearch_ingest.ensure_index, "chunks"),
        )

        # -- index + backup (Qdrant, OpenSearch, S3 chunks.jsonl — parallel)
        await _log_stage("index_and_backup_chunks")
        chunks_jsonl = "\n".join(
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
            for c, db in zip(chunks, db_chunks, strict=True)
        ).encode()

        await asyncio.gather(
            asyncio.to_thread(
                qdrant_ingest.upsert_chunks,
                "documents",
                document_id,
                chunks_with_vectors,
                user_id=user_id,
            ),
            asyncio.to_thread(
                opensearch_ingest.bulk_index,
                "chunks",
                document_id,
                os_chunks,
                user_id=user_id,
            ),
            s3_client.upload_bytes(
                f"{base_key}/chunks.jsonl",
                chunks_jsonl,
                "application/jsonl",
                bucket=get_s3_chunks_bucket(),
            ),
        )

        await asyncio.gather(
            asyncio.to_thread(qdrant_ingest.delete_by_chunk_ids, "documents", old_chunk_ids),
            asyncio.to_thread(opensearch_ingest.bulk_delete, "chunks", old_chunk_ids),
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
            await _emit("error", {"message": str(exc)})
        except Exception:
            logger.exception("pipeline.set_failed_error", extra={"document_id": document_id})
        raise
    finally:
        if pdf_path is not None:
            pdf_path.unlink(missing_ok=True)


@celery_app.task(bind=True, name="ingest_document")
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
