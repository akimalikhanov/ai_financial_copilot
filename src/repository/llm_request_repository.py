from __future__ import annotations

from decimal import Decimal
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.llm_request import LLMRequest


class LLMRequestRepository:
    """Repository for LLM request CRUD operations."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        conversation_id: UUID,
        user_id: UUID | None,
        provider: str,
        model: str,
        request_params: dict | None = None,
    ) -> LLMRequest:
        """Create a new LLM request record."""
        llm_request = LLMRequest(
            conversation_id=conversation_id,
            user_id=user_id,
            provider=provider,
            model=model,
            request_params=request_params or {},
        )
        self.session.add(llm_request)
        await self.session.flush()
        return llm_request

    async def update_on_final(
        self,
        request_id: UUID,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        reasoning_tokens: int | None = None,
        total_tokens: int | None = None,
        cost_usd: Decimal | None = None,
        latency_ms: int | None = None,
        ttft_ms: int | None = None,
        tps: int | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> LLMRequest | None:
        """Update LLM request stats on stream completion."""
        result = await self.session.execute(select(LLMRequest).where(LLMRequest.id == request_id))
        llm_request = result.scalar_one_or_none()
        if not llm_request:
            return None

        llm_request.prompt_tokens = prompt_tokens
        llm_request.completion_tokens = completion_tokens
        llm_request.reasoning_tokens = reasoning_tokens
        llm_request.total_tokens = total_tokens
        llm_request.cost_usd = cost_usd
        llm_request.latency_ms = latency_ms
        llm_request.ttft_ms = ttft_ms
        llm_request.tps = tps
        llm_request.error_code = error_code
        llm_request.error_message = error_message

        await self.session.flush()
        return llm_request
