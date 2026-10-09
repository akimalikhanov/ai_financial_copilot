from __future__ import annotations

from collections.abc import Sequence

from src.schemas.chat import Turn
from src.services.context.turns import answer_history, as_messages, cap_turns
from src.services.llm_adapters.base_adapter import ChatMessage as AdapterChatMessage
from src.services.llm_adapters.base_adapter import Role as AdapterRole
from src.services.prompts.prompt_renderer import PromptRenderer


def assemble_prompt(
    prior: Sequence[Turn],
    system_prompt: str,
    rag_context: str,
    user_query: str,
    renderer: PromptRenderer,
) -> list[AdapterChatMessage]:
    """System prompt, prior turns capped to `answer_history()`, then the current question
    rendered with its context."""
    rendered_user = renderer.render_user_message(
        context=rag_context,
        user_query=user_query,
    )
    return [
        AdapterChatMessage(role=AdapterRole.system, content=system_prompt),
        *as_messages(cap_turns(prior, answer_history())),
        AdapterChatMessage(role=AdapterRole.user, content=rendered_user),
    ]
