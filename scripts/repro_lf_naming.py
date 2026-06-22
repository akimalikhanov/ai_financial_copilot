"""Repro/regression: first-message naming generation must NOT cause trace to vanish.

Mirrors the span-stack structure of src/services/chat/tasks.py:
  _lf_stack:   chat_pipeline(root, trace_context) -> propagate_attributes
  _gen_stack:  chat_model(gen) — closed explicitly after .update(), before naming
  _stage_stack: per-stage span, closed/reopened by _log_stage
  naming: a standalone start_as_current_observation(generation) like RoutedLLM.complete

Bug (fixed): chat_model was in _lf_stack. _log_stage("persist_and_emit") closed
stream_llm_response via _stage_stack.close(), resetting the contextvar token BEFORE
chat_model's token — corrupting the observation chain for naming and the root trace.

Runs two pipelines with distinct trace_ids:
  - NO_NAMING  (mimics msg #2)
  - WITH_NAMING (mimics msg #1, seq==2)
Both should now appear in Langfuse.
"""

from __future__ import annotations

import base64
import contextlib
import os
import time
from uuid import UUID, uuid4

import httpx
from langfuse import propagate_attributes

from src.observability import langfuse as lf_client


def run_pipeline(*, request_id: str, conversation_id: str, with_naming: bool) -> None:
    lf = lf_client.get_client()
    assert lf is not None, "langfuse client not initialized"

    _lf_stack = contextlib.ExitStack()
    _stage_stack: contextlib.ExitStack = contextlib.ExitStack()
    _gen_stack: contextlib.ExitStack = contextlib.ExitStack()

    def log_stage(name: str, as_type: str = "span") -> None:
        nonlocal _stage_stack
        _stage_stack.close()
        _stage_stack = contextlib.ExitStack()
        _stage_stack.enter_context(
            lf.start_as_current_observation(as_type=as_type, name=name)  # type: ignore[arg-type]
        )

    try:
        # root
        _lf_stack.enter_context(
            lf.start_as_current_observation(
                as_type="chain",
                name="chat_pipeline",
                trace_context={"trace_id": UUID(request_id).hex},
                input={"request_id": request_id},
            )
        )
        # propagate_attributes (session = conversation)
        _lf_stack.enter_context(
            propagate_attributes(
                user_id="repro-user",
                session_id=conversation_id,
                metadata={"request_id": request_id},
            )
        )

        log_stage("route_query", "chain")
        log_stage("build_rag_context", "retriever")
        log_stage("render_prompt", "span")

        # chat_model gen uses its own stack so it can be closed before naming,
        # while still inside the persist_and_emit stage.
        log_stage("stream_llm_response", "span")
        gen = _gen_stack.enter_context(
            lf.start_as_current_observation(
                as_type="generation", name="chat_model", model="gpt-4o", input=[{"role": "u"}]
            )
        )

        # persist stage: stream stage span closes here, gen stays open a moment longer
        log_stage("persist_and_emit", "span")

        # update and close chat_model gen BEFORE naming fires
        gen.update(output="final answer text")  # type: ignore[union-attr]
        _gen_stack.close()

        # naming: standalone generation (RoutedLLM.complete pattern), seq==2 only
        if with_naming:
            with lf.start_as_current_observation(
                as_type="generation", name="conversation_naming", model="gpt-4o-mini"
            ) as g:
                g.update(output="A Title")
    finally:
        _stage_stack.close()
        _gen_stack.close()  # no-op if already closed
        _lf_stack.close()


def fetch_traces(host: str, pk: str, sk: str, ids: list[str]) -> dict[str, bool]:
    auth = base64.b64encode(f"{pk}:{sk}".encode()).decode()
    out: dict[str, bool] = {}
    with httpx.Client(base_url=host, headers={"Authorization": f"Basic {auth}"}, timeout=10) as c:
        for tid in ids:
            r = c.get(f"/api/public/traces/{tid}")
            out[tid] = r.status_code == 200
    return out


def main() -> None:
    lf_client.initialize()
    cfg = {
        "host": os.environ["LANGFUSE_HOST"],
        "pk": os.environ["LANGFUSE_PUBLIC_KEY"],
        "sk": os.environ["LANGFUSE_SECRET_KEY"],
    }
    conv = str(uuid4())
    no_naming = str(uuid4())
    with_naming = str(uuid4())

    print(f"conversation_id = {conv}")
    print(f"NO_NAMING   request_id={no_naming}  trace_id={UUID(no_naming).hex}")
    print(f"WITH_NAMING request_id={with_naming} trace_id={UUID(with_naming).hex}")

    run_pipeline(request_id=no_naming, conversation_id=conv, with_naming=False)
    lf_client.flush()
    run_pipeline(request_id=with_naming, conversation_id=conv, with_naming=True)
    lf_client.flush()

    print("flushed; waiting 6s for ingestion...")
    time.sleep(6)
    present = fetch_traces(
        cfg["host"], cfg["pk"], cfg["sk"], [UUID(no_naming).hex, UUID(with_naming).hex]
    )
    print("\n=== RESULTS (trace present in Langfuse?) ===")
    print(f"NO_NAMING   {UUID(no_naming).hex}: {present[UUID(no_naming).hex]}")
    print(f"WITH_NAMING {UUID(with_naming).hex}: {present[UUID(with_naming).hex]}")


if __name__ == "__main__":
    main()
