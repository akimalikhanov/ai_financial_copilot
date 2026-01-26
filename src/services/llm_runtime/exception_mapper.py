"""Maps provider-specific exceptions to internal exception types."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.services.llm_runtime.exceptions import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMContentFilterError,
    LLMContextLengthError,
    LLMError,
    LLMInvalidRequestError,
    LLMNotFoundError,
    LLMPermissionError,
    LLMRateLimitError,
    LLMServerError,
    LLMServiceUnavailableError,
    LLMTimeoutError,
    RetryInfo,
)

if TYPE_CHECKING:
    pass


def _extract_retry_after(error: Exception) -> float | None:
    """Extract Retry-After value from error headers if available."""
    # OpenAI errors have response.headers
    response = getattr(error, "response", None)
    if response is not None:
        headers = getattr(response, "headers", {})
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass
    return None


def map_openai_error(
    error: Exception,
    *,
    provider: str = "openai",
    model: str | None = None,
) -> LLMError:
    """Map OpenAI SDK exceptions to internal exception types."""
    from openai import (
        APIConnectionError,
        APIStatusError,
        APITimeoutError,
        AuthenticationError,
        BadRequestError,
        InternalServerError,
        NotFoundError,
        PermissionDeniedError,
        RateLimitError,
    )

    retry_after = _extract_retry_after(error)
    retry_info = RetryInfo(retry_after_seconds=retry_after) if retry_after else None

    common_kwargs = {
        "provider": provider,
        "model": model,
        "original_error": error,
        "retry_info": retry_info,
    }

    if isinstance(error, RateLimitError):
        return LLMRateLimitError(str(error), **common_kwargs)

    if isinstance(error, AuthenticationError):
        return LLMAuthenticationError(str(error), **common_kwargs)

    if isinstance(error, PermissionDeniedError):
        return LLMPermissionError(str(error), **common_kwargs)

    if isinstance(error, NotFoundError):
        return LLMNotFoundError(str(error), **common_kwargs)

    if isinstance(error, BadRequestError):
        # Check for specific error types
        msg = str(error).lower()
        if "context_length" in msg or "maximum context" in msg or "too long" in msg:
            return LLMContextLengthError(str(error), **common_kwargs)
        if "content" in msg and ("filter" in msg or "policy" in msg or "blocked" in msg):
            return LLMContentFilterError(str(error), **common_kwargs)
        return LLMInvalidRequestError(str(error), **common_kwargs)

    if isinstance(error, InternalServerError):
        # Check status code for more specific mapping
        status_code = getattr(error, "status_code", 500)
        if status_code == 503:
            return LLMServiceUnavailableError(str(error), **common_kwargs)
        return LLMServerError(str(error), **common_kwargs)

    if isinstance(error, APITimeoutError):
        return LLMTimeoutError(str(error), **common_kwargs)

    if isinstance(error, APIConnectionError):
        return LLMConnectionError(str(error), **common_kwargs)

    if isinstance(error, APIStatusError):
        # Fallback for other status errors
        status_code = getattr(error, "status_code", None)
        if status_code == 429:
            return LLMRateLimitError(str(error), **common_kwargs)
        if status_code in (500, 502):
            return LLMServerError(str(error), **common_kwargs)
        if status_code == 503:
            return LLMServiceUnavailableError(str(error), **common_kwargs)
        return LLMError(str(error), **common_kwargs)

    # Unknown error - wrap as generic LLMError
    return LLMError(str(error), **common_kwargs)


def map_google_error(
    error: Exception,
    *,
    provider: str = "google",
    model: str | None = None,
) -> LLMError:
    """Map Google GenAI SDK exceptions to internal exception types."""
    # google.genai uses google.api_core.exceptions
    from google.api_core import exceptions as google_exceptions

    common_kwargs = {
        "provider": provider,
        "model": model,
        "original_error": error,
    }

    if isinstance(error, google_exceptions.ResourceExhausted):
        return LLMRateLimitError(str(error), **common_kwargs)

    if isinstance(error, google_exceptions.Unauthenticated):
        return LLMAuthenticationError(str(error), **common_kwargs)

    if isinstance(error, google_exceptions.PermissionDenied):
        return LLMPermissionError(str(error), **common_kwargs)

    if isinstance(error, google_exceptions.NotFound):
        return LLMNotFoundError(str(error), **common_kwargs)

    if isinstance(error, google_exceptions.InvalidArgument):
        msg = str(error).lower()
        if "token" in msg and ("limit" in msg or "exceed" in msg):
            return LLMContextLengthError(str(error), **common_kwargs)
        return LLMInvalidRequestError(str(error), **common_kwargs)

    if isinstance(error, google_exceptions.DeadlineExceeded):
        return LLMTimeoutError(str(error), **common_kwargs)

    if isinstance(error, google_exceptions.ServiceUnavailable):
        return LLMServiceUnavailableError(str(error), **common_kwargs)

    if isinstance(error, google_exceptions.InternalServerError):
        return LLMServerError(str(error), **common_kwargs)

    if isinstance(error, google_exceptions.GoogleAPIError):
        return LLMError(str(error), **common_kwargs)

    # Unknown error
    return LLMError(str(error), **common_kwargs)
