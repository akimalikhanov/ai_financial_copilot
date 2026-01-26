"""
Unit tests for LLM exception hierarchy and exception mappers.

Tests:
1. Exception hierarchy and class attributes (is_retryable, default_retry_delay)
2. Exception metadata (provider, model, original_error, retry_info)
3. OpenAI error mapping
4. Google error mapping
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.services.llm_runtime.exception_mapper import (
    _extract_retry_after,
    map_google_error,
    map_openai_error,
)
from src.services.llm_runtime.exceptions import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMContentFilterError,
    LLMContextLengthError,
    LLMError,
    LLMInvalidRequestError,
    LLMNotFoundError,
    LLMPermanentError,
    LLMPermissionError,
    LLMQuotaExceededError,
    LLMRateLimitError,
    LLMServerError,
    LLMServiceUnavailableError,
    LLMTimeoutError,
    LLMTransientError,
    RetryInfo,
)


# =========================================================
# Test Exception Hierarchy
# =========================================================
class TestExceptionHierarchy:
    """Test that exception classes have correct hierarchy and attributes."""

    def test_base_llm_error_defaults(self):
        """LLMError should have is_retryable=False by default."""
        error = LLMError("test error")
        assert error.is_retryable is False
        assert error.default_retry_delay == 1.0
        assert str(error) == "test error"

    def test_llm_error_with_metadata(self):
        """LLMError should store all metadata correctly."""
        original = ValueError("original")
        retry_info = RetryInfo(retry_after_seconds=30.0, attempt=2)

        error = LLMError(
            "test error",
            provider="openai",
            model="gpt-4",
            original_error=original,
            retry_info=retry_info,
            details={"key": "value"},
        )

        assert error.provider == "openai"
        assert error.model == "gpt-4"
        assert error.original_error is original
        assert error.retry_info is retry_info
        assert error.retry_info is not None
        assert error.retry_info.retry_after_seconds == 30.0
        assert error.retry_info.attempt == 2
        assert error.details == {"key": "value"}

    def test_transient_errors_are_retryable(self):
        """All transient errors should have is_retryable=True."""
        transient_errors = [
            LLMTransientError("test"),
            LLMRateLimitError("rate limit"),
            LLMServiceUnavailableError("unavailable"),
            LLMTimeoutError("timeout"),
            LLMConnectionError("connection"),
            LLMServerError("server error"),
        ]

        for error in transient_errors:
            assert error.is_retryable is True, f"{type(error).__name__} should be retryable"
            assert isinstance(error, LLMTransientError)
            assert isinstance(error, LLMError)

    def test_permanent_errors_not_retryable(self):
        """All permanent errors should have is_retryable=False."""
        permanent_errors = [
            LLMPermanentError("test"),
            LLMAuthenticationError("auth"),
            LLMPermissionError("permission"),
            LLMInvalidRequestError("invalid"),
            LLMNotFoundError("not found"),
            LLMContentFilterError("content filter"),
            LLMContextLengthError("context length"),
            LLMQuotaExceededError("quota"),
        ]

        for error in permanent_errors:
            assert error.is_retryable is False, f"{type(error).__name__} should NOT be retryable"
            assert isinstance(error, LLMPermanentError)
            assert isinstance(error, LLMError)

    def test_rate_limit_has_longer_default_delay(self):
        """LLMRateLimitError should have a longer default delay."""
        error = LLMRateLimitError("rate limited")
        assert error.default_retry_delay == 5.0

    def test_server_errors_have_medium_delay(self):
        """Server errors should have medium default delay."""
        assert LLMServerError("err").default_retry_delay == 2.0
        assert LLMServiceUnavailableError("err").default_retry_delay == 2.0

    def test_connection_timeout_have_short_delay(self):
        """Connection and timeout errors should have short default delay."""
        assert LLMConnectionError("err").default_retry_delay == 1.0
        assert LLMTimeoutError("err").default_retry_delay == 1.0


class TestRetryInfo:
    """Test RetryInfo dataclass."""

    def test_retry_info_defaults(self):
        info = RetryInfo()
        assert info.retry_after_seconds is None
        assert info.attempt == 0
        assert info.max_attempts == 3

    def test_retry_info_with_values(self):
        info = RetryInfo(retry_after_seconds=10.5, attempt=2, max_attempts=5)
        assert info.retry_after_seconds == 10.5
        assert info.attempt == 2
        assert info.max_attempts == 5

    def test_retry_info_is_frozen(self):
        info = RetryInfo(retry_after_seconds=10.0)
        with pytest.raises(AttributeError):
            info.retry_after_seconds = 20.0  # type: ignore


# =========================================================
# Test OpenAI Error Mapping
# =========================================================
class TestOpenAIErrorMapping:
    """Test map_openai_error function."""

    def test_rate_limit_error(self):
        """RateLimitError should map to LLMRateLimitError."""
        from openai import RateLimitError

        # Create a mock RateLimitError
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {}

        error = RateLimitError(
            "Rate limit exceeded",
            response=mock_response,
            body=None,
        )

        result = map_openai_error(error, provider="openai", model="gpt-4")

        assert isinstance(result, LLMRateLimitError)
        assert result.is_retryable is True
        assert result.provider == "openai"
        assert result.model == "gpt-4"
        assert result.original_error is error

    def test_authentication_error(self):
        """AuthenticationError should map to LLMAuthenticationError."""
        from openai import AuthenticationError

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.headers = {}

        error = AuthenticationError(
            "Invalid API key",
            response=mock_response,
            body=None,
        )

        result = map_openai_error(error, provider="openai", model="gpt-4")

        assert isinstance(result, LLMAuthenticationError)
        assert result.is_retryable is False

    def test_not_found_error(self):
        """NotFoundError should map to LLMNotFoundError."""
        from openai import NotFoundError

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.headers = {}

        error = NotFoundError(
            "Model not found",
            response=mock_response,
            body=None,
        )

        result = map_openai_error(error, provider="openai", model="gpt-999")

        assert isinstance(result, LLMNotFoundError)
        assert result.is_retryable is False
        assert result.model == "gpt-999"

    def test_bad_request_error_generic(self):
        """BadRequestError should map to LLMInvalidRequestError."""
        from openai import BadRequestError

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.headers = {}

        error = BadRequestError(
            "Invalid parameters",
            response=mock_response,
            body=None,
        )

        result = map_openai_error(error, provider="openai", model="gpt-4")

        assert isinstance(result, LLMInvalidRequestError)
        assert result.is_retryable is False

    def test_bad_request_context_length(self):
        """BadRequestError with context_length message should map to LLMContextLengthError."""
        from openai import BadRequestError

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.headers = {}

        error = BadRequestError(
            "This model's maximum context length is 8192 tokens",
            response=mock_response,
            body=None,
        )

        result = map_openai_error(error, provider="openai", model="gpt-4")

        assert isinstance(result, LLMContextLengthError)

    def test_bad_request_content_filter(self):
        """BadRequestError with content filter message should map to LLMContentFilterError."""
        from openai import BadRequestError

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.headers = {}

        error = BadRequestError(
            "Your request was blocked due to content policy violation",
            response=mock_response,
            body=None,
        )

        result = map_openai_error(error, provider="openai", model="gpt-4")

        assert isinstance(result, LLMContentFilterError)

    def test_permission_denied_error(self):
        """PermissionDeniedError should map to LLMPermissionError."""
        from openai import PermissionDeniedError

        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.headers = {}

        error = PermissionDeniedError(
            "Permission denied",
            response=mock_response,
            body=None,
        )

        result = map_openai_error(error, provider="openai", model="gpt-4")

        assert isinstance(result, LLMPermissionError)
        assert result.is_retryable is False

    def test_internal_server_error(self):
        """InternalServerError should map to LLMServerError."""
        from openai import InternalServerError

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.headers = {}

        error = InternalServerError(
            "Internal server error",
            response=mock_response,
            body=None,
        )

        result = map_openai_error(error, provider="openai", model="gpt-4")

        assert isinstance(result, LLMServerError)
        assert result.is_retryable is True

    def test_internal_server_error_503(self):
        """InternalServerError with 503 should map to LLMServiceUnavailableError."""
        from openai import InternalServerError

        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.headers = {}

        error = InternalServerError(
            "Service unavailable",
            response=mock_response,
            body=None,
        )
        error.status_code = 503

        result = map_openai_error(error, provider="openai", model="gpt-4")

        assert isinstance(result, LLMServiceUnavailableError)
        assert result.is_retryable is True

    def test_api_timeout_error(self):
        """APITimeoutError should map to LLMTimeoutError."""
        from openai import APITimeoutError

        error = APITimeoutError(request=MagicMock())

        result = map_openai_error(error, provider="openai", model="gpt-4")

        assert isinstance(result, LLMTimeoutError)
        assert result.is_retryable is True

    def test_api_connection_error(self):
        """APIConnectionError should map to LLMConnectionError."""
        from openai import APIConnectionError

        error = APIConnectionError(request=MagicMock())

        result = map_openai_error(error, provider="openai", model="gpt-4")

        assert isinstance(result, LLMConnectionError)
        assert result.is_retryable is True

    def test_unknown_error_maps_to_base(self):
        """Unknown errors should map to base LLMError."""
        error = ValueError("some random error")

        result = map_openai_error(error, provider="openai", model="gpt-4")

        assert type(result) is LLMError
        assert result.original_error is error

    def test_vllm_provider_name(self):
        """Provider name should be correctly passed through."""
        from openai import APIConnectionError

        error = APIConnectionError(request=MagicMock())

        result = map_openai_error(error, provider="vllm", model="llama-3")

        assert result.provider == "vllm"
        assert result.model == "llama-3"


class TestExtractRetryAfter:
    """Test _extract_retry_after helper function."""

    def test_extracts_retry_after_header(self):
        """Should extract Retry-After header value."""
        mock_response = MagicMock()
        mock_response.headers = {"retry-after": "30"}

        error = MagicMock()
        error.response = mock_response

        result = _extract_retry_after(error)
        assert result == 30.0

    def test_extracts_retry_after_header_capitalized(self):
        """Should handle capitalized Retry-After header."""
        mock_response = MagicMock()
        mock_response.headers = {"Retry-After": "45"}

        error = MagicMock()
        error.response = mock_response

        result = _extract_retry_after(error)
        assert result == 45.0

    def test_returns_none_when_no_header(self):
        """Should return None when no Retry-After header."""
        mock_response = MagicMock()
        mock_response.headers = {}

        error = MagicMock()
        error.response = mock_response

        result = _extract_retry_after(error)
        assert result is None

    def test_returns_none_when_no_response(self):
        """Should return None when error has no response."""
        error = ValueError("test")
        result = _extract_retry_after(error)
        assert result is None

    def test_handles_invalid_retry_after_value(self):
        """Should return None for non-numeric Retry-After."""
        mock_response = MagicMock()
        mock_response.headers = {"retry-after": "invalid"}

        error = MagicMock()
        error.response = mock_response

        result = _extract_retry_after(error)
        assert result is None


# =========================================================
# Test Google Error Mapping
# =========================================================
class TestGoogleErrorMapping:
    """Test map_google_error function."""

    def test_resource_exhausted_error(self):
        """ResourceExhausted should map to LLMRateLimitError."""
        from google.api_core import exceptions as google_exceptions

        error = google_exceptions.ResourceExhausted("Quota exceeded")

        result = map_google_error(error, provider="google", model="gemini-pro")

        assert isinstance(result, LLMRateLimitError)
        assert result.is_retryable is True
        assert result.provider == "google"
        assert result.model == "gemini-pro"

    def test_unauthenticated_error(self):
        """Unauthenticated should map to LLMAuthenticationError."""
        from google.api_core import exceptions as google_exceptions

        error = google_exceptions.Unauthenticated("Invalid API key")

        result = map_google_error(error, provider="google", model="gemini-pro")

        assert isinstance(result, LLMAuthenticationError)
        assert result.is_retryable is False

    def test_permission_denied_error(self):
        """PermissionDenied should map to LLMPermissionError."""
        from google.api_core import exceptions as google_exceptions

        error = google_exceptions.PermissionDenied("Access denied")

        result = map_google_error(error, provider="google", model="gemini-pro")

        assert isinstance(result, LLMPermissionError)
        assert result.is_retryable is False

    def test_not_found_error(self):
        """NotFound should map to LLMNotFoundError."""
        from google.api_core import exceptions as google_exceptions

        error = google_exceptions.NotFound("Model not found")

        result = map_google_error(error, provider="google", model="gemini-999")

        assert isinstance(result, LLMNotFoundError)
        assert result.is_retryable is False

    def test_invalid_argument_error(self):
        """InvalidArgument should map to LLMInvalidRequestError."""
        from google.api_core import exceptions as google_exceptions

        error = google_exceptions.InvalidArgument("Invalid parameter")

        result = map_google_error(error, provider="google", model="gemini-pro")

        assert isinstance(result, LLMInvalidRequestError)
        assert result.is_retryable is False

    def test_invalid_argument_token_limit(self):
        """InvalidArgument with token limit message should map to LLMContextLengthError."""
        from google.api_core import exceptions as google_exceptions

        error = google_exceptions.InvalidArgument("Token limit exceeded")

        result = map_google_error(error, provider="google", model="gemini-pro")

        assert isinstance(result, LLMContextLengthError)

    def test_deadline_exceeded_error(self):
        """DeadlineExceeded should map to LLMTimeoutError."""
        from google.api_core import exceptions as google_exceptions

        error = google_exceptions.DeadlineExceeded("Request timeout")

        result = map_google_error(error, provider="google", model="gemini-pro")

        assert isinstance(result, LLMTimeoutError)
        assert result.is_retryable is True

    def test_service_unavailable_error(self):
        """ServiceUnavailable should map to LLMServiceUnavailableError."""
        from google.api_core import exceptions as google_exceptions

        error = google_exceptions.ServiceUnavailable("Service down")

        result = map_google_error(error, provider="google", model="gemini-pro")

        assert isinstance(result, LLMServiceUnavailableError)
        assert result.is_retryable is True

    def test_internal_server_error(self):
        """InternalServerError should map to LLMServerError."""
        from google.api_core import exceptions as google_exceptions

        error = google_exceptions.InternalServerError("Internal error")

        result = map_google_error(error, provider="google", model="gemini-pro")

        assert isinstance(result, LLMServerError)
        assert result.is_retryable is True

    def test_generic_google_api_error(self):
        """Generic GoogleAPIError should map to base LLMError."""
        from google.api_core import exceptions as google_exceptions

        error = google_exceptions.GoogleAPIError("Some error")

        result = map_google_error(error, provider="google", model="gemini-pro")

        assert type(result) is LLMError
        assert result.original_error is error

    def test_unknown_error_maps_to_base(self):
        """Unknown errors should map to base LLMError."""
        error = ValueError("some random error")

        result = map_google_error(error, provider="google", model="gemini-pro")

        assert type(result) is LLMError
        assert result.original_error is error
