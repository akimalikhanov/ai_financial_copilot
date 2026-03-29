from __future__ import annotations

from typing import Any

from jinja2 import Environment, StrictUndefined, TemplateSyntaxError, Undefined, UndefinedError
from jinja2.sandbox import SandboxedEnvironment

from src.schemas.prompt import PromptTemplate
from src.services.prompts.prompt_loader import PromptLoader, get_prompt_loader


class PromptRendererError(Exception):
    """Raised when prompt rendering fails."""

    pass


class PromptRenderer:
    """
    Render prompt templates using Jinja2.

    Usage:
        renderer = PromptRenderer()
        system_prompt = get_system_prompt()
        user_message = renderer.render_user_message(
            context="Retrieved document excerpts...",
            user_query="What was Apple's revenue?"
        )
    """

    def __init__(
        self,
        loader: PromptLoader | None = None,
        *,
        strict: bool = True,
        sandboxed: bool = True,
    ):
        """
        Initialize the prompt renderer.

        Args:
            loader: PromptLoader instance. If None, uses the default cached loader.
            strict: If True, raise errors for undefined variables. Defaults to True.
            sandboxed: If True, use sandboxed Jinja2 environment for security.
                Defaults to True.
        """
        self._loader = loader or get_prompt_loader()

        # Create Jinja2 environment
        env_class = SandboxedEnvironment if sandboxed else Environment
        self._env = env_class(
            # Use StrictUndefined to catch missing variables, or Undefined for lenient mode
            undefined=StrictUndefined if strict else Undefined,
            # Keep whitespace handling predictable
            trim_blocks=True,
            lstrip_blocks=True,
            # Don't auto-escape (we're not rendering HTML)
            autoescape=False,
        )

    @property
    def loader(self) -> PromptLoader:
        """Return the underlying prompt loader."""
        return self._loader

    def render_user_message(
        self,
        context: str,
        user_query: str,
        *,
        version: str = "v1",
    ) -> str:
        """
        Render a user message using the user template.

        Args:
            context: Retrieved document excerpts.
            user_query: The user's query text.
            version: User template version. Defaults to "v1".

        Returns:
            Rendered user message.

        Raises:
            PromptRendererError: If rendering fails.
        """
        # Load the user template
        template = self._loader.load("user", version)

        # Validate required variables
        self._validate_variables(template, {"context": context, "user_query": user_query})

        # Render the template
        return self._render_template(
            template.template, {"context": context, "user_query": user_query}
        )

    def _render_template(self, template_string: str, variables: dict[str, Any]) -> str:
        """
        Internal method to render a template string.

        Args:
            template_string: Jinja2 template string.
            variables: Variables to substitute.

        Returns:
            Rendered string.

        Raises:
            PromptRendererError: If rendering fails.
        """
        try:
            jinja_template = self._env.from_string(template_string)
            return jinja_template.render(**variables)
        except TemplateSyntaxError as e:
            raise PromptRendererError(f"Template syntax error: {e}") from e
        except UndefinedError as e:
            raise PromptRendererError(f"Undefined variable in template: {e}") from e
        except Exception as e:
            raise PromptRendererError(f"Template rendering failed: {e}") from e

    def _validate_variables(self, template: PromptTemplate, variables: dict[str, Any]) -> None:
        """
        Validate that all required variables are provided.

        Args:
            template: PromptTemplate with required variables list.
            variables: Provided variables.

        Raises:
            PromptRendererError: If required variables are missing.
        """
        required = set(template.variables)
        provided = set(variables.keys())
        missing = required - provided

        if missing:
            raise PromptRendererError(
                f"Missing required variables for template '{template.name}': {sorted(missing)}"
            )


def get_system_prompt(version: str = "v1") -> str:
    """
    Get the system prompt by loading the system template.

    Args:
        version: System template version. Defaults to "v1".

    Returns:
        System prompt string.

    Raises:
        PromptRendererError: If loading fails.
    """
    loader = get_prompt_loader()
    template = loader.load("system", version)
    # System template has no variables, so just return the template as-is
    return template.template


_renderer: PromptRenderer | None = None


def get_prompt_renderer(loader: PromptLoader | None = None) -> PromptRenderer:
    """
    Get the cached PromptRenderer singleton (default loader).

    Args:
        loader: Optional PromptLoader. If None, uses the default cached loader.
            Passing a custom loader bypasses the singleton (test/override use only).

    Returns:
        PromptRenderer instance.
    """
    global _renderer
    if loader is not None:
        return PromptRenderer(loader)
    if _renderer is None:
        _renderer = PromptRenderer()
    return _renderer
