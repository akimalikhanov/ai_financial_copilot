from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from src.schemas.prompt import PromptTemplate
from src.utils.config import get_project_root


class PromptLoaderError(Exception):
    """Raised when prompt loading fails."""

    pass


def _get_default_prompts_dir() -> Path:
    """Get the default prompts directory."""
    return get_project_root() / "prompts"


class PromptLoader:
    """
    Load and manage prompt templates from YAML files.

    Templates are loaded from YAML files with filename pattern: {name}_{version}.yaml
    For example: system_v1.yaml, persona_v2.yaml

    Usage:
        loader = PromptLoader()
        template = loader.load("system", "v1")
        print(template.template)
    """

    def __init__(self, prompts_dir: Path | str | None = None):
        """
        Initialize the prompt loader.

        Args:
            prompts_dir: Directory containing prompt YAML files.
                If None, uses the default prompts/ directory in project root.
        """
        if prompts_dir is None:
            self._prompts_dir = _get_default_prompts_dir()
        else:
            self._prompts_dir = Path(prompts_dir)

        if not self._prompts_dir.exists():
            raise PromptLoaderError(f"Prompts directory does not exist: {self._prompts_dir}")

    @property
    def prompts_dir(self) -> Path:
        """Return the prompts directory path."""
        return self._prompts_dir

    def _get_template_path(self, name: str, version: str) -> Path:
        """
        Get the path to a template file.

        Args:
            name: Template name (e.g., "system", "persona")
            version: Version string (e.g., "v1", "v2")

        Returns:
            Path to the template file.
        """
        filename = f"{name}_{version}.yaml"
        return self._prompts_dir / filename

    def load(self, name: str, version: str = "v1") -> PromptTemplate:
        """
        Load a prompt template by name and version.

        Args:
            name: Template name (e.g., "system", "persona", "guidelines", "user")
            version: Version string (e.g., "v1", "v2"). Defaults to "v1".

        Returns:
            PromptTemplate object with the loaded template.

        Raises:
            PromptLoaderError: If the template file doesn't exist or is invalid.
        """
        template_path = self._get_template_path(name, version)

        if not template_path.exists():
            available = self.list_versions(name)
            if available:
                raise PromptLoaderError(
                    f"Template '{name}' version '{version}' not found. "
                    f"Available versions: {available}"
                )
            raise PromptLoaderError(
                f"Template '{name}' not found. Available templates: {self.list_templates()}"
            )

        try:
            with open(template_path) as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise PromptLoaderError(f"Invalid YAML in {template_path}: {e}") from e

        if not isinstance(data, dict):
            raise PromptLoaderError(f"Template file must contain a YAML mapping: {template_path}")

        try:
            return PromptTemplate(**data)
        except Exception as e:
            raise PromptLoaderError(f"Invalid template structure in {template_path}: {e}") from e

    def list_versions(self, name: str) -> list[str]:
        """
        List all available versions for a template name.

        Args:
            name: Template name (e.g., "system", "persona")

        Returns:
            List of version strings (e.g., ["v1", "v2"]), sorted.
        """
        pattern = f"{name}_*.yaml"
        versions = []

        for path in self._prompts_dir.glob(pattern):
            # Extract version from filename: {name}_{version}.yaml
            match = re.match(rf"^{re.escape(name)}_(.+)\.yaml$", path.name)
            if match:
                versions.append(match.group(1))

        return sorted(versions)

    def list_templates(self) -> list[str]:
        """
        List all available template names.

        Returns:
            List of unique template names (e.g., ["system", "persona", "guidelines"]).
        """
        names = set()

        for path in self._prompts_dir.glob("*.yaml"):
            # Extract name from filename: {name}_{version}.yaml
            match = re.match(r"^(.+)_[^_]+\.yaml$", path.name)
            if match:
                names.add(match.group(1))

        return sorted(names)

    def get_latest_version(self, name: str) -> str:
        """
        Get the latest version string for a template name.

        Args:
            name: Template name (e.g., "system", "persona")

        Returns:
            Latest version string (e.g., "v2").

        Raises:
            PromptLoaderError: If no versions exist for the template.
        """
        versions = self.list_versions(name)
        if not versions:
            raise PromptLoaderError(
                f"No versions found for template '{name}'. "
                f"Available templates: {self.list_templates()}"
            )
        # Sort versions naturally (v1, v2, v10, etc.)
        return sorted(versions, key=_version_sort_key)[-1]

    def load_latest(self, name: str) -> PromptTemplate:
        """
        Load the latest version of a template.

        Args:
            name: Template name (e.g., "system", "persona")

        Returns:
            PromptTemplate object with the latest version.

        Raises:
            PromptLoaderError: If the template doesn't exist.
        """
        version = self.get_latest_version(name)
        return self.load(name, version)

    def load_raw(self, name: str, version: str = "v1") -> dict[str, Any]:
        """
        Load raw YAML data without validation.

        Useful for debugging or when you need access to the raw data.

        Args:
            name: Template name
            version: Version string

        Returns:
            Raw dictionary from YAML file.
        """
        template_path = self._get_template_path(name, version)

        if not template_path.exists():
            raise PromptLoaderError(f"Template file not found: {template_path}")

        with open(template_path) as f:
            return yaml.safe_load(f)


def _version_sort_key(version: str) -> tuple[int, str]:
    """
    Sort key for version strings.

    Handles versions like "v1", "v2", "v10" correctly.
    """
    match = re.match(r"^v(\d+)$", version)
    if match:
        return (int(match.group(1)), "")
    return (0, version)


@lru_cache(maxsize=1)
def get_prompt_loader(prompts_dir: str | None = None) -> PromptLoader:
    """
    Get a cached PromptLoader instance.

    Args:
        prompts_dir: Optional path to prompts directory.
            If None, uses the default prompts/ directory.

    Returns:
        Cached PromptLoader instance.
    """
    return PromptLoader(prompts_dir)
