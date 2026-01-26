"""Standardized exception hierarchy for LLM operations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RetryInfo:
    """Metadata for retry behavior."""

    retry_after_seconds: float | None = None  # From Retry-After header
    attempt: int = 0  # Current attempt number
    max_attempts: int = 3  # Default max attempts


class LLMError(Exception):
    """Base exception for all LLM-related errors."""

    is_retryable: bool = False
    default_retry_delay: float = 1.0  # seconds

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        original_error: Exception | None = None,
        retry_info: RetryInfo | None = None,
        details: dict[str, Any] | None = None,
        error_code: str | None = None,
        status_code: int | None = None,
    ):
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.original_error = original_error
        self.retry_info = retry_info
        self.details = details or {}
        self.error_code = error_code
        self.status_code = status_code

    def to_dict(self, *, as_json: bool = False, indent: int = 2) -> dict[str, Any] | str:
        """
        Convert exception to a readable dictionary format or JSON string.

        Args:
            as_json: If True, return JSON string instead of dict (default: False)
            indent: Number of spaces for JSON indentation when as_json=True (default: 2)

        Returns:
            Dictionary with error details if as_json=False, otherwise JSON string.
            Dict keys: error_type, message, provider, model, is_retryable,
            error_code, status_code, details, retry_info, original_error_type

        Example:
            >>> error.to_dict()  # Returns dict
            >>> error.to_dict(as_json=True)  # Returns formatted JSON string
        """
        result: dict[str, Any] = {
            "error_type": self.__class__.__name__,
            "message": str(self),
            "is_retryable": self.is_retryable,
        }

        if self.provider:
            result["provider"] = self.provider

        if self.model:
            result["model"] = self.model

        if self.error_code:
            result["error_code"] = self.error_code

        if self.status_code:
            result["status_code"] = self.status_code

        if self.details:
            result["details"] = self.details

        if self.retry_info:
            result["retry_info"] = {
                "retry_after_seconds": self.retry_info.retry_after_seconds,
                "attempt": self.retry_info.attempt,
                "max_attempts": self.retry_info.max_attempts,
            }

        if self.original_error:
            result["original_error_type"] = type(self.original_error).__name__
            # result["original_error_message"] = str(self.original_error)

        if as_json:
            return json.dumps(result, indent=indent, default=str)

        return result

    def __repr__(self) -> str:
        """Provide a concise repr showing key error details."""
        parts = [f"{self.__class__.__name__}"]

        if self.error_code:
            parts.append(f"code={self.error_code}")

        if self.status_code:
            parts.append(f"status={self.status_code}")

        if self.provider:
            parts.append(f"provider={self.provider}")

        if self.model:
            parts.append(f"model={self.model}")

        # Truncate message if too long
        msg = str(self)
        if len(msg) > 100:
            msg = msg[:97] + "..."
        parts.append(f"msg='{msg}'")

        return f"{parts[0]}({', '.join(parts[1:])})"


# ============ Retryable (Transient) Errors ============


class LLMTransientError(LLMError):
    """Base class for transient, retryable errors."""

    is_retryable: bool = True


class LLMRateLimitError(LLMTransientError):
    """Rate limit exceeded (HTTP 429). Retryable with backoff."""

    default_retry_delay: float = 5.0


class LLMServiceUnavailableError(LLMTransientError):
    """Service temporarily unavailable (HTTP 503). Retryable."""

    default_retry_delay: float = 2.0


class LLMTimeoutError(LLMTransientError):
    """Request timeout (HTTP 408/504). Retryable."""

    default_retry_delay: float = 1.0


class LLMConnectionError(LLMTransientError):
    """Network connection error. Retryable."""

    default_retry_delay: float = 1.0


class LLMServerError(LLMTransientError):
    """Internal server error (HTTP 500/502). Retryable with caution."""

    default_retry_delay: float = 2.0


# ============ Non-Retryable (Permanent) Errors ============


class LLMPermanentError(LLMError):
    """Base class for permanent, non-retryable errors."""

    is_retryable: bool = False


class LLMAuthenticationError(LLMPermanentError):
    """Invalid or missing API key (HTTP 401)."""

    pass


class LLMPermissionError(LLMPermanentError):
    """Permission denied / forbidden (HTTP 403)."""

    pass


class LLMInvalidRequestError(LLMPermanentError):
    """Invalid request parameters (HTTP 400/422)."""

    pass


class LLMNotFoundError(LLMPermanentError):
    """Model or resource not found (HTTP 404)."""

    pass


class LLMContentFilterError(LLMPermanentError):
    """Content blocked by safety filters."""

    pass


class LLMContextLengthError(LLMPermanentError):
    """Input exceeds model's context length."""

    pass


class LLMQuotaExceededError(LLMPermanentError):
    """Account quota exceeded (different from rate limit)."""

    pass
