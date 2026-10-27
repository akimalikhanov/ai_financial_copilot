"""Integration test: sample chats search the demo user's corpus, and nothing else does.

Real API, DB and chat pipeline (Celery eager, mock LLMs). The mock tool model makes one
search; a spy on the retrieval call records which owner and documents it searched. The
owner filter inside Qdrant/OpenSearch is covered by test_retrieval_clients.py.

Requires: PostgreSQL (via PgBouncer) and Redis (redis-app, redis-broker) running.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import AsyncGenerator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest
import yaml
from httpx import AsyncClient
from sqlalchemy import delete

from src.db.connection import get_session_factory
from src.models.document import Document
from src.models.user import User
from src.schemas.retrieval import RetrievalTrace
from src.services.chat import tasks
from src.services.chat.agent import loop
from src.services.llm_adapters.base_adapter import AssistantTurnResult, ToolCallRef
from src.services.router.company_name import normalize_company
from src.services.sample.manifest import load_manifest
from tests.integration.conftest import TOOL_MODEL_ID

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


class _SearchOnceToolLLM:
    """Tool model that searches once, then reports nothing."""

    provider = "mock"
    api = "mock"
    model_id = TOOL_MODEL_ID

    def __init__(self) -> None:
        self._searched = False

    async def close(self) -> None:
        pass

    async def complete_with_tools(self, *_args: Any, **_kwargs: Any) -> AssistantTurnResult:
        if self._searched:
            call = ToolCallRef(id="c2", name="report_findings", arguments='{"findings": []}')
        else:
            self._searched = True
            args = json.dumps({"entity": "", "query": "net loss", "keywords": "net loss"})
            call = ToolCallRef(id="c1", name="search_documents", arguments=args)
        return AssistantTurnResult(text="", tool_calls=[call])


async def _register(client: AsyncClient) -> tuple[UUID, dict[str, str]]:
    resp = await client.post(
        "/v1/auth/register",
        json={"email": f"sample-{uuid4().hex}@test.com", "password": "testpass123"},
    )
    assert resp.status_code == 200
    headers = {"Authorization": f"Bearer {resp.json()['access_token']}"}
    me = await client.get("/v1/auth/me", headers=headers)
    assert me.status_code == 200
    return UUID(me.json()["id"]), headers


def _doc(doc_id: UUID, user_id: UUID, company: str) -> Document:
    return Document(
        id=doc_id,
        user_id=user_id,
        original_filename=f"{company}.pdf",
        storage_key=f"uploads/{doc_id}.pdf",
        status="ready",
        document_metadata={"company": company, "year": "2022"},
        company_norm=normalize_company(company),
    )


@pytest.fixture
async def world(
    async_client: AsyncClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> AsyncGenerator[SimpleNamespace, None]:
    """A demo user owning one sample doc, user X owning one doc, and user Y owning none."""
    demo_id, sample_doc, x_doc = uuid4(), uuid4(), uuid4()
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "documents": [
                    {"id": str(sample_doc), "file": "s.pdf", "sha256": "0", "title": "Sample"}
                ],
                "questions": [{"id": "q", "text": "What was the net loss?"}],
            }
        )
    )
    monkeypatch.setenv("DEMO_USER_ID", str(demo_id))
    monkeypatch.setenv("SAMPLE_MANIFEST_PATH", str(manifest))
    load_manifest.cache_clear()

    x_id, x_headers = await _register(async_client)
    _, y_headers = await _register(async_client)
    async with get_session_factory()() as session:
        session.add(User(id=demo_id, email=f"demo-{demo_id.hex}@sample.invalid", is_active=False))
        await session.flush()
        session.add_all([_doc(sample_doc, demo_id, "Sample Co"), _doc(x_doc, x_id, "Acme Corp")])
        await session.commit()

    searches: list[tuple[UUID, list[UUID] | None]] = []

    async def _spy(_session: Any, *, user_id: UUID, doc_ids: list[UUID] | None, **_: Any):
        searches.append((user_id, doc_ids))
        return None, RetrievalTrace(), []

    monkeypatch.setattr(loop, "run_chat_rag_pipeline", _spy)
    models = tasks._router._models  # type: ignore[union-attr]
    monkeypatch.setitem(
        models,
        TOOL_MODEL_ID,
        dataclasses.replace(models[TOOL_MODEL_ID], adapter=_SearchOnceToolLLM()),
    )

    yield SimpleNamespace(
        demo_id=demo_id,
        sample_doc=sample_doc,
        x_id=x_id,
        x_doc=x_doc,
        x_headers=x_headers,
        y_headers=y_headers,
        searches=searches,
    )

    async with get_session_factory()() as session:
        await session.execute(delete(Document).where(Document.id.in_([sample_doc, x_doc])))
        await session.execute(delete(User).where(User.id == demo_id))
        await session.commit()
    load_manifest.cache_clear()


async def _chat(
    client: AsyncClient, headers: dict[str, str], conversation_id: str, scope: dict | None = None
) -> int:
    resp = await client.post(
        "/v1/chat",
        json={
            "conversation_id": conversation_id,
            "content": "What was the net loss?",
            "client_msg_id": str(uuid4()),
            "client_request_id": str(uuid4()),
            "model": "gpt-4o-mini",
            "params": {},
            "metadata": {"scope": scope} if scope else {},
        },
        headers=headers,
    )
    return resp.status_code


async def _sample_conversation(client: AsyncClient, headers: dict[str, str]) -> str:
    resp = await client.post("/v1/samples/conversations", headers=headers)
    assert resp.status_code == 200
    return resp.json()["conversation_id"]


async def test_sample_chat_searches_demo_corpus_only(async_client, world) -> None:
    """A sample chat searches the demo docs, even when the client scope names the user's own."""
    conv = await _sample_conversation(async_client, world.x_headers)
    own_doc = {"mode": "selectedDocs", "doc_ids": [str(world.x_doc)]}

    assert await _chat(async_client, world.x_headers, conv, scope=own_doc) == 200

    assert world.searches == [(world.demo_id, [world.sample_doc])]


async def test_normal_chat_never_sees_demo_corpus(async_client, world) -> None:
    resp = await async_client.post("/v1/conversations", json={}, headers=world.x_headers)
    conv = resp.json()["conversation_id"]

    assert await _chat(async_client, world.x_headers, conv, scope={"mode": "allDocs"}) == 200

    assert world.searches == [(world.x_id, [world.x_doc])]


async def test_other_user_cannot_reach_user_docs_through_sample_paths(async_client, world) -> None:
    # Y's sample chat, scoped to X's document, still searches only the demo corpus.
    y_conv = await _sample_conversation(async_client, world.y_headers)
    x_doc_scope = {"mode": "selectedDocs", "doc_ids": [str(world.x_doc)]}
    assert await _chat(async_client, world.y_headers, y_conv, scope=x_doc_scope) == 200
    assert world.searches == [(world.demo_id, [world.sample_doc])]

    # Y can't post into X's sample conversation.
    x_conv = await _sample_conversation(async_client, world.x_headers)
    assert await _chat(async_client, world.y_headers, x_conv) == 404

    # The PDF endpoint's demo exception doesn't extend to X's documents.
    pdf = await async_client.get(f"/v1/documents/{world.x_doc}/pdf", headers=world.y_headers)
    assert pdf.status_code == 404


async def test_client_cannot_flag_a_conversation_as_sample(async_client, world) -> None:
    resp = await async_client.post(
        "/v1/conversations", json={"settings": {"corpus": "demo"}}, headers=world.y_headers
    )
    assert resp.status_code == 200
    listed = await async_client.get("/v1/conversations", headers=world.y_headers)
    flags = {c["id"]: c["is_sample"] for c in listed.json()["conversations"]}
    assert flags[resp.json()["conversation_id"]] is False
