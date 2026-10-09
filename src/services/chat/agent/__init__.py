"""Agentic RAG: the tool-calling loop plus the synthesis boundary, as one call.

Public API: ``run_agent`` + ``AgentRunResult``. Both callers (`tasks.py`,
`src.eval.pipeline_agent`) collapse to a single call, so production and eval cannot drift.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.services.chat.agent.loop import run_loop
from src.services.chat.agent.state import AgentLoopMeta, AgentSettings, get_agent_settings
from src.services.chat.agent.synthesis import AgentRunResult, run_synthesis
from src.utils.config import get_agent_fallback_max_chunks

if TYPE_CHECKING:
    from collections.abc import Sequence

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
    *,
    fallbacks: Sequence[RoutedLLM] = (),
) -> AgentRunResult:
    """Run the tool-calling loop, then synthesize its output. One call, one boundary."""
    evidence, findings, meta = await run_loop(
        state, llm, session, redis_app, request_id, reranker, session_factory, fallbacks=fallbacks
    )
    requested_currency = getattr(state.router_output, "requested_currency", None)
    return await run_synthesis(
        evidence,
        findings,
        meta,
        requested_currency,
        fallback_max_chunks=get_agent_fallback_max_chunks(),
        mentions=state.scope_result.mentions() if state.scope_result else None,
    )
