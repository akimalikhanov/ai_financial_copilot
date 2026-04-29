from __future__ import annotations

from decimal import Decimal
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.llm_request import LLMRequest
from src.models.message import Message, MessageRole, MessageStatus
from src.services.llm_adapters.base_adapter import LLMResponseStats


def stats_to_request_kwargs(stats: LLMResponseStats | None) -> dict:
    """Map adapter LLMResponseStats → llm_requests column kwargs (handles type coercion)."""
    if stats is None:
        return {}
    return {
        "prompt_tokens": stats.input_tokens,
        "completion_tokens": stats.output_tokens,
        "reasoning_tokens": stats.reasoning_tokens,
        "total_tokens": stats.total_tokens,
        "cost_usd": Decimal(str(stats.cost_usd)) if stats.cost_usd is not None else None,
        "latency_ms": int(stats.latency_ms) if stats.latency_ms is not None else None,
        "ttft_ms": int(stats.ttft_ms) if stats.ttft_ms is not None else None,
        "tps": int(stats.tps) if stats.tps is not None else None,
    }


class LLMRequestRepository:
    """Repository for LLM request CRUD operations."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_id(self, request_id: UUID) -> LLMRequest | None:
        """Get LLM request by id."""
        result = await self.session.execute(select(LLMRequest).where(LLMRequest.id == request_id))
        return result.scalar_one_or_none()

    async def list_recent_by_user_id(self, user_id: UUID, limit: int = 50) -> list[LLMRequest]:
        """List top-level chat requests for user, newest first. Excludes sub-requests."""
        result = await self.session.execute(
            select(LLMRequest)
            .where(LLMRequest.user_id == user_id)
            .where(LLMRequest.parent_request_id.is_(None))
            .order_by(LLMRequest.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def list_recent_by_conversation_id(
        self, conversation_id: UUID, limit: int = 50
    ) -> list[LLMRequest]:
        """List top-level completed chat requests for a conversation. Excludes sub-requests."""
        result = await self.session.execute(
            select(LLMRequest)
            .where(LLMRequest.conversation_id == conversation_id)
            .where(LLMRequest.status == "completed")
            .where(LLMRequest.parent_request_id.is_(None))
            .order_by(LLMRequest.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_by_client_request_id(
        self, conversation_id: UUID, client_request_id: str
    ) -> LLMRequest | None:
        """Get LLM request by client_request_id for idempotency check."""
        result = await self.session.execute(
            select(LLMRequest).where(
                LLMRequest.conversation_id == conversation_id,
                LLMRequest.client_request_id == client_request_id,
            )
        )
        return result.scalar_one_or_none()

    async def update_status(self, request_id: UUID, status: str) -> LLMRequest | None:
        """Update LLM request status."""
        result = await self.session.execute(select(LLMRequest).where(LLMRequest.id == request_id))
        llm_request = result.scalar_one_or_none()
        if not llm_request:
            return None

        llm_request.status = status
        await self.session.flush()
        return llm_request

    async def create_with_placeholder(
        self,
        conversation_id: UUID,
        user_id: UUID | None,
        provider: str,
        model: str,
        user_message_id: UUID | None = None,
        snapshot_seq: int | None = None,
        client_request_id: str | None = None,
        included_message_ids: list[UUID] | None = None,
        request_params: dict | None = None,
        initial_status: str = "pending",
    ) -> tuple[LLMRequest, Message]:
        """Atomically create LLM request with assistant message placeholder."""
        from sqlalchemy import func

        # Get next seq for assistant message
        result = await self.session.execute(
            select(func.coalesce(func.max(Message.seq), 0)).where(
                Message.conversation_id == conversation_id
            )
        )
        next_seq = result.scalar_one() + 1

        # Use in_progress for placeholder; queued requests start as in_progress
        # (message_status enum has no 'queued', avoiding DB migration)
        placeholder_status = MessageStatus.in_progress

        # Create assistant placeholder
        assistant_message = Message(
            conversation_id=conversation_id,
            user_id=user_id,
            role=MessageRole.assistant,
            content="",
            seq=next_seq,
            status=placeholder_status,
            message_metadata={},
        )
        self.session.add(assistant_message)
        await self.session.flush()

        # Create LLM request linked to placeholder
        # Convert UUID list to string list for JSON serialization
        included_message_ids_json = (
            [str(msg_id) for msg_id in included_message_ids] if included_message_ids else None
        )
        llm_request = LLMRequest(
            conversation_id=conversation_id,
            user_id=user_id,
            provider=provider,
            model=model,
            user_message_id=user_message_id,
            snapshot_seq=snapshot_seq,
            client_request_id=client_request_id,
            included_message_ids=included_message_ids_json,
            assistant_message_id=assistant_message.id,
            status=initial_status,
            request_params=request_params or {},
        )
        self.session.add(llm_request)
        await self.session.flush()

        # Link assistant message to request
        assistant_message.request_id = llm_request.id
        await self.session.flush()

        return llm_request, assistant_message

    async def create_subrequest(
        self,
        parent_request_id: UUID,
        conversation_id: UUID,
        user_id: UUID | None,
        provider: str,
        model: str,
        request_type: str,
        request_params: dict | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        reasoning_tokens: int | None = None,
        total_tokens: int | None = None,
        cost_usd: Decimal | None = None,
        latency_ms: int | None = None,
        ttft_ms: int | None = None,
        tps: int | None = None,
        status: str = "completed",
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> LLMRequest:
        """Create a completed sub-request row linked to a parent chat request."""
        sub = LLMRequest(
            conversation_id=conversation_id,
            user_id=user_id,
            provider=provider,
            model=model,
            parent_request_id=parent_request_id,
            request_type=request_type,
            request_params=request_params or {},
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            tps=tps,
            status=status,
            error_code=error_code,
            error_message=error_message,
        )
        self.session.add(sub)
        await self.session.flush()
        return sub

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
        trace_id: str | None = None,
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
        if trace_id is not None:
            llm_request.trace_id = trace_id

        await self.session.flush()
        return llm_request
