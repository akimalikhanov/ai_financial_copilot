"""Shared Celery app for ingestion and chat tasks."""

from __future__ import annotations

import os

from celery import Celery

from src.utils.config import get_redis_broker_url


def _get_int_env(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    return int(raw)


celery_app = Celery(
    "ai_copilot",
    broker=get_redis_broker_url(),
    include=[
        "src.services.ingestion.tasks",
        "src.workers.chat_worker",
    ],
)

_task_soft_limit = _get_int_env("CELERY_TASK_SOFT_TIME_LIMIT_SECONDS")
_task_limit = _get_int_env("CELERY_TASK_TIME_LIMIT_SECONDS")

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    worker_hijack_root_logger=False,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_track_started=True,
    broker_transport_options={
        "visibility_timeout": int(os.getenv("CELERY_VISIBILITY_TIMEOUT_SECONDS", "7200"))
    },
    task_soft_time_limit=_task_soft_limit,
    task_time_limit=_task_limit,
    task_routes={
        "process_chat": {"queue": "chat"},
        "ingest_document": {"queue": "ingestion"},
    },
)
