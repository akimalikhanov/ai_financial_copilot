"""
Unit tests for prompt loading and rendering logic.

Tests:
1. PromptLoader - loading templates, listing versions, error handling
2. PromptRenderer - rendering user messages, getting system prompt
3. Error cases - missing variables, invalid templates, missing files
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest
import yaml

from src.schemas.prompt import PromptTemplate
from src.services.prompts.prompt_loader import PromptLoader, PromptLoaderError
from src.services.prompts.prompt_renderer import (
    PromptRenderer,
    PromptRendererError,
    get_system_prompt,
)

# -------------------------
# Test fixtures
# -------------------------


@pytest.fixture
def temp_prompts_dir():
    """Create a temporary directory with test prompt templates."""
    with TemporaryDirectory() as tmpdir:
        prompts_dir = Path(tmpdir)

        # Create system_v1.yaml
        system_template = {
            "version": "v1",
            "name": "system",
            "description": "Test system prompt",
            "template": "You are a test assistant.\n",
            "variables": [],
        }
        (prompts_dir / "system_v1.yaml").write_text(yaml.dump(system_template))

        # Create user_v1.yaml
        user_template = {
            "version": "v1",
            "name": "user",
            "description": "Test user prompt",
            "template": "Context: {{ context }}\nQuestion: {{ user_query }}",
            "variables": ["context", "user_query"],
        }
        (prompts_dir / "user_v1.yaml").write_text(yaml.dump(user_template))

        # Create user_v2.yaml (for version testing)
        user_v2_template = {
            "version": "v2",
            "name": "user",
            "description": "Test user prompt v2",
            "template": "**Context**: {{ context }}\n**Query**: {{ user_query }}",
            "variables": ["context", "user_query"],
        }
        (prompts_dir / "user_v2.yaml").write_text(yaml.dump(user_v2_template))

        yield prompts_dir


# -------------------------
# PromptLoader tests
# -------------------------


def test_prompt_loader_loads_template(temp_prompts_dir):
    """Test that PromptLoader can load a template."""
    loader = PromptLoader(temp_prompts_dir)
    template = loader.load("system", "v1")

    assert isinstance(template, PromptTemplate)
    assert template.name == "system"
    assert template.version == "v1"
    assert template.template == "You are a test assistant.\n"
    assert template.variables == []


def test_prompt_loader_loads_template_with_variables(temp_prompts_dir):
    """Test that PromptLoader loads templates with variables correctly."""
    loader = PromptLoader(temp_prompts_dir)
    template = loader.load("user", "v1")

    assert template.name == "user"
    assert template.version == "v1"
    assert "{{ context }}" in template.template
    assert "{{ user_query }}" in template.template
    assert set(template.variables) == {"context", "user_query"}


def test_prompt_loader_raises_on_missing_template(temp_prompts_dir):
    """Test that PromptLoader raises error for missing template."""
    loader = PromptLoader(temp_prompts_dir)

    with pytest.raises(PromptLoaderError, match="Template 'nonexistent' not found"):
        loader.load("nonexistent", "v1")


def test_prompt_loader_raises_on_missing_version(temp_prompts_dir):
    """Test that PromptLoader raises error for missing version."""
    loader = PromptLoader(temp_prompts_dir)

    with pytest.raises(PromptLoaderError, match="version 'v99' not found"):
        loader.load("system", "v99")


def test_prompt_loader_raises_on_invalid_yaml(temp_prompts_dir):
    """Test that PromptLoader raises error for invalid YAML."""
    # Create invalid YAML file
    invalid_file = temp_prompts_dir / "invalid_v1.yaml"
    invalid_file.write_text("invalid: yaml: content: [unclosed")

    loader = PromptLoader(temp_prompts_dir)

    with pytest.raises(PromptLoaderError, match="Invalid YAML"):
        loader.load("invalid", "v1")


def test_prompt_loader_list_versions(temp_prompts_dir):
    """Test that PromptLoader can list available versions."""
    loader = PromptLoader(temp_prompts_dir)

    versions = loader.list_versions("user")
    assert versions == ["v1", "v2"]

    versions = loader.list_versions("system")
    assert versions == ["v1"]


def test_prompt_loader_list_templates(temp_prompts_dir):
    """Test that PromptLoader can list available templates."""
    loader = PromptLoader(temp_prompts_dir)

    templates = loader.list_templates()
    assert set(templates) == {"system", "user"}


def test_prompt_loader_get_latest_version(temp_prompts_dir):
    """Test that PromptLoader can get the latest version."""
    loader = PromptLoader(temp_prompts_dir)

    latest = loader.get_latest_version("user")
    assert latest == "v2"

    latest = loader.get_latest_version("system")
    assert latest == "v1"


def test_prompt_loader_load_latest(temp_prompts_dir):
    """Test that PromptLoader can load the latest version."""
    loader = PromptLoader(temp_prompts_dir)

    template = loader.load_latest("user")
    assert template.version == "v2"
    assert "**Context**" in template.template


def test_prompt_loader_raises_on_missing_directory():
    """Test that PromptLoader raises error for missing directory."""
    with pytest.raises(PromptLoaderError, match="Prompts directory does not exist"):
        PromptLoader(Path("/nonexistent/directory"))


# -------------------------
# PromptRenderer tests
# -------------------------


def test_prompt_renderer_renders_user_message(temp_prompts_dir):
    """Test that PromptRenderer can render a user message."""
    loader = PromptLoader(temp_prompts_dir)
    renderer = PromptRenderer(loader)

    result = renderer.render_user_message(
        context="Document: Test Report\nPage 1: Revenue was $100M",
        user_query="What was the revenue?",
    )

    assert "Document: Test Report" in result
    assert "What was the revenue?" in result
    assert "Context:" in result
    assert "Question:" in result


def test_prompt_renderer_validates_required_variables():
    """Test that PromptRenderer validates required variables match template."""
    # Create a template with an extra required variable
    with TemporaryDirectory() as tmpdir:
        prompts_dir = Path(tmpdir)

        template_with_extra_var = {
            "version": "v1",
            "name": "user",
            "template": "Context: {{ context }}\nQuery: {{ user_query }}\nExtra: {{ extra_var }}",
            "variables": ["context", "user_query", "extra_var"],
        }
        (prompts_dir / "user_v1.yaml").write_text(yaml.dump(template_with_extra_var))

        loader = PromptLoader(prompts_dir)
        renderer = PromptRenderer(loader)

        # Should raise error because extra_var is required but not provided
        with pytest.raises(PromptRendererError, match="Missing required variables"):
            renderer.render_user_message(
                context="Some context",
                user_query="Some query",
            )


def test_prompt_renderer_raises_on_invalid_template_syntax():
    """Test that PromptRenderer raises error for invalid Jinja2 syntax."""
    # Create a temporary directory with invalid template
    with TemporaryDirectory() as tmpdir:
        prompts_dir = Path(tmpdir)

        # Create user template with invalid Jinja2 syntax
        invalid_template = {
            "version": "v1",
            "name": "user",
            "template": "Context: {{ context }}\nQuery: {{ unclosed",
            "variables": ["context", "user_query"],
        }
        (prompts_dir / "user_v1.yaml").write_text(yaml.dump(invalid_template))

        loader = PromptLoader(prompts_dir)
        renderer = PromptRenderer(loader)

        # Try to render it (will fail when Jinja2 tries to parse)
        with pytest.raises(PromptRendererError, match="Template syntax error"):
            renderer.render_user_message(
                context="test",
                user_query="test",
            )


def test_prompt_renderer_handles_empty_context(temp_prompts_dir):
    """Test that PromptRenderer handles empty context."""
    loader = PromptLoader(temp_prompts_dir)
    renderer = PromptRenderer(loader)

    result = renderer.render_user_message(
        context="",
        user_query="What is the answer?",
    )

    assert "Question:" in result
    assert "What is the answer?" in result


def test_prompt_renderer_handles_special_characters(temp_prompts_dir):
    """Test that PromptRenderer handles special characters in context/query."""
    loader = PromptLoader(temp_prompts_dir)
    renderer = PromptRenderer(loader)

    context = "Document: Test & Co.\nRevenue: $1,234.56 (up 5.5%)"
    query = "What's the revenue? (with details)"

    result = renderer.render_user_message(context=context, user_query=query)

    assert "Test & Co." in result
    assert "$1,234.56" in result
    assert "What's the revenue?" in result


# -------------------------
# get_system_prompt tests
# -------------------------


def test_get_system_prompt_loads_template(temp_prompts_dir):
    """Test that get_system_prompt loads the system template."""
    with patch("src.services.prompts.prompt_renderer.get_prompt_loader") as mock_loader:
        mock_loader_instance = PromptLoader(temp_prompts_dir)
        mock_loader.return_value = mock_loader_instance

        result = get_system_prompt()

        assert result == "You are a test assistant.\n"


def test_get_system_prompt_uses_default_version(temp_prompts_dir):
    """Test that get_system_prompt uses v1 by default."""
    with patch("src.services.prompts.prompt_renderer.get_prompt_loader") as mock_loader:
        mock_loader_instance = PromptLoader(temp_prompts_dir)
        mock_loader.return_value = mock_loader_instance

        result = get_system_prompt()

        # Should load v1
        assert result == "You are a test assistant.\n"


def test_get_system_prompt_uses_specified_version(temp_prompts_dir):
    """Test that get_system_prompt can use a specific version."""
    # Create system_v2.yaml
    system_v2_template = {
        "version": "v2",
        "name": "system",
        "description": "Test system prompt v2",
        "template": "You are a test assistant v2.\n",
        "variables": [],
    }
    (temp_prompts_dir / "system_v2.yaml").write_text(yaml.dump(system_v2_template))

    with patch("src.services.prompts.prompt_renderer.get_prompt_loader") as mock_loader:
        mock_loader_instance = PromptLoader(temp_prompts_dir)
        mock_loader.return_value = mock_loader_instance

        result = get_system_prompt(version="v2")

        assert result == "You are a test assistant v2.\n"


def test_full_flow_load_and_render(temp_prompts_dir):
    """Test the full flow of loading and rendering prompts."""
    loader = PromptLoader(temp_prompts_dir)
    renderer = PromptRenderer(loader)

    # Get system prompt using the temp loader
    with patch("src.services.prompts.prompt_renderer.get_prompt_loader") as mock_loader:
        mock_loader.return_value = loader
        system_prompt = get_system_prompt()
        assert "test assistant" in system_prompt.lower()

    # Render user message
    user_message = renderer.render_user_message(
        context="Document: Annual Report 2024\nPage 1: Revenue $500M",
        user_query="What was the revenue in 2024?",
    )

    assert "Annual Report 2024" in user_message
    assert "$500M" in user_message
    assert "What was the revenue in 2024?" in user_message


def test_renderer_with_default_loader():
    """Test that PromptRenderer works with default loader (real prompts directory)."""
    renderer = PromptRenderer()

    # Should be able to render user message with real templates
    result = renderer.render_user_message(
        context="Test context",
        user_query="Test query",
    )

    assert "Test context" in result
    assert "Test query" in result


def test_system_prompt_with_real_templates():
    """Test that get_system_prompt works with real templates."""
    # This uses the actual prompts directory
    system_prompt = get_system_prompt()

    # Should contain the Financial Document Analyst prompt
    assert "Financial Document Analyst" in system_prompt
    assert "RAG-based" in system_prompt or "Retrieval-Augmented Generation" in system_prompt
