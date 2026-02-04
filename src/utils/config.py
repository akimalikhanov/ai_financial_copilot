from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Load environment variables from .env file (if present)
load_dotenv()


def get_project_root() -> Path:
    """
    Get the project root directory.

    Uses PROJECT_ROOT environment variable if set, otherwise searches
    for pyproject.toml in parent directories.

    Returns:
        Path to the project root directory.

    Raises:
        RuntimeError: If project root cannot be determined.
    """
    if root := os.getenv("PROJECT_ROOT"):
        return Path(root)

    # Fallback: search for pyproject.toml
    current = Path(__file__).resolve()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent

    raise RuntimeError("Could not determine project root")


def _expand_env_vars(value: Any) -> Any:
    """Recursively expand ${VAR:-default} patterns in YAML values."""
    if isinstance(value, str):
        # Match ${VAR:-default} pattern
        pattern = r"\$\{([^:}]+)(?::-([^}]*))?\}"

        def replacer(match: re.Match[str]) -> str:
            var_name = match.group(1)
            default = match.group(2) if match.group(2) is not None else ""
            return os.getenv(var_name, default)

        return re.sub(pattern, replacer, value)
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    return value


def load_yaml_config(
    relative_path: str,
    *,
    config_path: Path | str | None = None,
    expand_env_vars: bool = False,
) -> dict[str, Any]:
    """
    Load a YAML config file from the project.

    Args:
        relative_path: Relative path from project root (e.g., "infra/config/models.yaml").
            Only used if config_path is None.
        config_path: Absolute path to config file. If provided, overrides relative_path.
        expand_env_vars: If True, expand ${VAR:-default} patterns in YAML values.

    Returns:
        Parsed YAML dict.
    """
    if config_path is None:
        project_root = get_project_root()
        config_path = project_root / relative_path
    else:
        config_path = Path(config_path)

    with open(config_path) as f:
        data = yaml.safe_load(f)

    if expand_env_vars:
        data = _expand_env_vars(data)

    return data


def load_models_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Load models.yaml config file with environment variable expansion.

    Args:
        config_path: Path to models.yaml. If None, uses infra/config/models.yaml relative to project root.

    Returns:
        Parsed YAML dict with env vars expanded.
    """
    if config_path is None:
        return load_yaml_config("infra/config/models.yaml", expand_env_vars=True)
    return load_yaml_config("", config_path=config_path, expand_env_vars=True)


def load_error_maps(config_path: Path | str | None = None) -> dict[int, dict[str, str]]:
    """
    Load error messages from error_maps.yaml config file.

    Args:
        config_path: Path to error_maps.yaml. If None, uses infra/config/error_maps.yaml
            relative to project root.

    Returns:
        Dictionary mapping status codes to dict with 'user' and 'internal' messages.
    """
    if config_path is None:
        data = load_yaml_config("infra/config/error_maps.yaml", expand_env_vars=False)
    else:
        data = load_yaml_config("", config_path=config_path, expand_env_vars=False)

    errors = data.get("errors", {})
    # Convert string keys to int keys
    return {int(k): v for k, v in errors.items()}


def get_cors_origins() -> list[str]:
    """
    Parse CORS_ALLOWED_ORIGINS from environment.

    CORS_ALLOWED_ORIGINS: Comma-separated list of allowed origins (required).
    Examples:
      - Development: "http://localhost:3000,http://127.0.0.1:3000"
      - Production:  "https://app.example.com,https://www.example.com"
      - Allow all (NOT recommended for production): "*"

    Raises:
        RuntimeError: If CORS_ALLOWED_ORIGINS is not set.
    """
    raw = os.getenv("CORS_ALLOWED_ORIGINS")
    if not raw:
        raise RuntimeError(
            "CORS_ALLOWED_ORIGINS environment variable is required. "
            "Set it to a comma-separated list of allowed origins "
            "(e.g., 'http://localhost:3000' for dev, or your production domain)."
        )
    if raw == "*":
        return ["*"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def get_db_url() -> str:
    """
    Build database connection URL from environment variables.

    Environment variables:
      - APP_DB_HOST: Database host (default: localhost)
      - APP_DB_PORT: Database port (default: 6432 for pgbouncer)
      - APP_DB_NAME: Database name (default: app)
      - APP_DB_USER: Database user (default: app)
      - APP_DB_PASSWORD: Database password (required)

    Returns:
        PostgreSQL async connection URL string.

    Raises:
        RuntimeError: If APP_DB_PASSWORD is not set.
    """
    host = os.getenv("APP_DB_HOST", "localhost")
    port = os.getenv("APP_DB_PORT", "6432")  # Default to pgbouncer port
    db_name = os.getenv("APP_DB_NAME", "app")
    db_user = os.getenv("APP_DB_USER", "app")
    db_password = os.getenv("APP_DB_PASSWORD")

    if not db_password:
        raise RuntimeError(
            "APP_DB_PASSWORD environment variable is required. "
            "Set it to the database password for the application database."
        )

    # Use asyncpg driver for async PostgreSQL connections
    return f"postgresql+asyncpg://{db_user}:{db_password}@{host}:{port}/{db_name}"
