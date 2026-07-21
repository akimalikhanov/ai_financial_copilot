"""Agentic RAG: the tool-calling loop plus the synthesis boundary, as one call.

Public API: ``run_agent`` + ``AgentRunResult``. Both callers (`tasks.py`,
`src.eval.pipeline_agent`) collapse to a single call and cannot drift (P0-5) —
see docs/stages/agentic_state_refactor_v2.md, *The boundary*.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.services.chat.agent.loop import run_loop
from src.services.chat.agent.state import AgentLoopMeta, AgentSettings, get_agent_settings
from src.services.chat.agent.synthesis import AgentRunResult, run_synthesis

if TYPE_CHECKING:
    from redis.asyncio import Redis
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from src.schemas.chat import ChatPipelineState
    from src.services.llm_router import RoutedLLM
    from src.services.retrieval.reranker import Reranker

__all__ = ["AgentLoopMeta", "AgentRunResult", "AgentSettings", "get_agent_settings", "run_agent"]


async def run_agent(
    state: ChatPipelineState,
    llm: RoutedLLM,
    session: AsyncSession,
    redis_app: Redis,
    request_id: str,
    reranker: Reranker | None,
    session_factory: async_sessionmaker[AsyncSession],
) -> AgentRunResult:
    """Run the tool-calling loop, then synthesize its output. One call, one boundary."""
    chunk_registry, findings, meta = await run_loop(
        state, llm, session, redis_app, request_id, reranker, session_factory
    )
    requested_currency = getattr(state.router_output, "requested_currency", None)
    return await run_synthesis(
        chunk_registry, findings, meta, state.scope_result, requested_currency, session
    )
