"""Shared circuit breaker and retry utilities for external service calls.

Uses tenacity for retry with exponential backoff, and a custom
CircuitBreaker class (tenacity 9.x removed the built-in one).

Usage::

    from src.services.circuit_breaker import async_circuit_breaker_retry

    @async_circuit_breaker_retry()
    async def call_external_service():
        ...
"""

import logging
import threading
import time
from enum import Enum
from functools import wraps
from typing import Callable, Optional, Tuple, Type

from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Simple circuit breaker: closed -> open after N failures -> half-open after timeout -> closed."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.monotonic() - self._last_failure_time >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
            return self._state

    def record_success(self) -> None:
        with self._lock:
            self._failure_count = 0
            self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(
                    f"Circuit breaker OPEN after {self._failure_count} failures (recovery in {self.recovery_timeout}s)"
                )

    def __enter__(self):
        if self.state == CircuitState.OPEN:
            raise CircuitBreakerOpenError(f"Circuit breaker is open (recovery in {self.recovery_timeout}s)")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.record_success()
        else:
            self.record_failure()
        return False


class CircuitBreakerOpenError(Exception):
    """Raised when a circuit breaker is in the open state."""


# Shared circuit breaker instances per service
_circuit_breakers: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
) -> CircuitBreaker:
    """Get or create a named circuit breaker instance."""
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
        )
    return _circuit_breakers[name]


def async_circuit_breaker_retry(
    service_name: str = "default",
    max_retries: int = 3,
    retry_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
    min_wait: float = 1.0,
    max_wait: float = 8.0,
) -> Callable:
    """Decorator that adds circuit breaker + retry with exponential backoff.

    Args:
        service_name: Name for the circuit breaker (shared per service)
        max_retries: Maximum number of retry attempts
        retry_exceptions: Tuple of exception types to retry on
        failure_threshold: Failures before circuit opens
        recovery_timeout: Seconds before half-open retry
        min_wait: Minimum wait time between retries (seconds)
        max_wait: Maximum wait time between retries (seconds)
    """
    cb = get_circuit_breaker(service_name, failure_threshold, recovery_timeout)

    def _log_retry(retry_state: RetryCallState):
        if retry_state.attempt_number > 1:
            exc = retry_state.outcome.exception() if retry_state.outcome else None
            logger.warning(f"[{service_name}] Retry {retry_state.attempt_number - 1}/{max_retries} after: {exc}")

    def decorator(func: Callable) -> Callable:
        retry_decorator = retry(
            stop=stop_after_attempt(max_retries + 1),
            wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
            retry=retry_if_exception_type(retry_exceptions),
            before=_log_retry,
            reraise=True,
        )

        @wraps(func)
        async def wrapper(*args, **kwargs):
            with cb:
                return await retry_decorator(func)(*args, **kwargs)

        wrapper.circuit_breaker = cb  # type: ignore[attr-defined]
        return wrapper

    return decorator


def async_retry_only(
    max_retries: int = 3,
    retry_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    min_wait: float = 1.0,
    max_wait: float = 8.0,
) -> Callable:
    """Lighter decorator: retry with exponential backoff, no circuit breaker."""

    def _log_retry(retry_state: RetryCallState):
        if retry_state.attempt_number > 1:
            exc = retry_state.outcome.exception() if retry_state.outcome else None
            logger.warning(f"Retry {retry_state.attempt_number - 1}/{max_retries} after: {exc}")

    def decorator(func: Callable) -> Callable:
        @retry(
            stop=stop_after_attempt(max_retries + 1),
            wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
            retry=retry_if_exception_type(retry_exceptions),
            before=_log_retry,
            reraise=True,
        )
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        return wrapper

    return decorator
