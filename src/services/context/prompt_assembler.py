from __future__ import annotations

from src.schemas import chat as schemas
from src.services.llm_adapters.base_adapter import ChatMessage as AdapterChatMessage
from src.services.llm_adapters.base_adapter import Role as AdapterRole
from src.services.prompts.prompt_renderer import PromptRenderer


def _to_adapter_messages(messages: list[schemas.ChatMessage]) -> list[AdapterChatMessage]:
    return [
        AdapterChatMessage(
            role=AdapterRole(m.role),
            content=m.content,
            name=m.name,
            tool_call_id=m.tool_call_id,
        )
        for m in messages
    ]


def assemble_prompt(
    history: list[schemas.ChatMessage],
    system_prompt: str,
    rag_context: str,
    user_query: str,
    renderer: PromptRenderer,
) -> list[AdapterChatMessage]:
    rendered_user = renderer.render_user_message(
        context=rag_context,
        user_query=user_query,
    )
    modified = list(history)
    if modified and modified[-1].role == schemas.Role.user:
        modified[-1] = schemas.ChatMessage(role=schemas.Role.user, content=rendered_user)
    else:
        modified.append(schemas.ChatMessage(role=schemas.Role.user, content=rendered_user))

    return [
        AdapterChatMessage(role=AdapterRole.system, content=system_prompt),
    ] + _to_adapter_messages(modified)
