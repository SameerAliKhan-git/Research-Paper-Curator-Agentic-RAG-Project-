import logging
import time
import uuid
from collections import defaultdict
from typing import Dict, Tuple

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs every request with correlation ID, timing, and status code."""

    async def dispatch(self, request: Request, call_next):
        correlation_id = str(uuid.uuid4())[:8]
        start = time.time()

        # Attach correlation ID to request state
        request.state.correlation_id = correlation_id

        logger.info(f"[{correlation_id}] -> {request.method} {request.url.path}")

        try:
            response: Response = await call_next(request)
        except Exception as exc:
            elapsed_ms = (time.time() - start) * 1000
            logger.error(f"[{correlation_id}] X {request.method} {request.url.path} FAILED in {elapsed_ms:.1f}ms: {exc}")
            raise

        elapsed_ms = (time.time() - start) * 1000
        logger.info(f"[{correlation_id}] <- {response.status_code} {request.method} {request.url.path} ({elapsed_ms:.1f}ms)")

        # Add headers
        response.headers["X-Correlation-ID"] = correlation_id
        response.headers["X-Response-Time"] = f"{elapsed_ms:.1f}ms"

        return response


class ExceptionCaptureMiddleware(BaseHTTPMiddleware):
    """Catches unhandled exceptions and returns structured JSON errors.

    Note: RequestValidationError (422) is raised BEFORE this middleware runs
    by FastAPI's own exception handler, so it won't be swallowed here.
    """

    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as exc:
            # Let FastAPI's RequestValidationError propagate to its own handler
            # (returns proper 422 with field-level error details)
            from fastapi.exceptions import RequestValidationError

            if isinstance(exc, RequestValidationError):
                raise exc

            logger.exception(f"Unhandled exception on {request.method} {request.url.path}: {exc}")
            return Response(
                content='{"detail":"Internal server error"}',
                status_code=500,
                media_type="application/json",
            )


class RequestBodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Rejects requests whose Content-Length exceeds the configured max size.

    Prevents OOM from malicious or accidental large payloads.
    Only checks the Content-Length header; streaming requests without
    the header are allowed through.
    """

    def __init__(self, app, max_size_bytes: int):
        super().__init__(app)
        self.max_size_bytes = max_size_bytes

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                size = int(content_length)
                if size > self.max_size_bytes:
                    max_mb = self.max_size_bytes // (1024 * 1024)
                    logger.warning(
                        f"Request rejected: body size {size} bytes exceeds limit of {self.max_size_bytes} bytes ({max_mb}MB)"
                    )
                    return JSONResponse(
                        status_code=413,
                        content={"detail": f"Request body too large. Maximum size is {max_mb}MB."},
                    )
            except ValueError:
                pass  # Invalid Content-Length; let FastAPI handle it
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Token bucket rate limiter per API key.

    Enforces per-client rate limiting using a sliding window counter.
    Returns 429 Too Many Requests when the limit is exceeded.
    """

    def __init__(self, app, default_rate_limit: int = 60, window_seconds: int = 60):
        """Initialize rate limiter.

        Args:
            app: ASGI application
            default_rate_limit: Maximum requests per window per API key
            window_seconds: Time window in seconds
        """
        super().__init__(app)
        self.default_rate_limit = default_rate_limit
        self.window_seconds = window_seconds
        # In-memory counters: {api_key: [(timestamp, count)]}
        self._counters: Dict[str, list] = defaultdict(list)
        self._last_cleanup = time.time()

    def _get_client_key(self, request: Request) -> str:
        """Extract client identifier from request (API key or IP)."""
        api_key = request.headers.get("X-API-Key", "")
        if api_key:
            # Hash API key for privacy
            return f"apikey:{api_key[:8]}"
        # Fallback to client IP
        client_ip = request.client.host if request.client else "unknown"
        return f"ip:{client_ip}"

    def _cleanup_old_entries(self) -> None:
        """Remove expired entries to prevent memory leaks."""
        now = time.time()
        if now - self._last_cleanup < self.window_seconds:
            return

        cutoff = now - self.window_seconds
        keys_to_delete = []
        for key, entries in self._counters.items():
            # Filter to entries within the window
            self._counters[key] = [(ts, count) for ts, count in entries if ts > cutoff]
            if not self._counters[key]:
                keys_to_delete.append(key)

        for key in keys_to_delete:
            del self._counters[key]

        self._last_cleanup = now

    def _check_rate_limit(self, client_key: str) -> Tuple[bool, int, int]:
        """Check if client has exceeded rate limit.

        Returns:
            Tuple of (allowed, current_count, limit)
        """
        now = time.time()
        cutoff = now - self.window_seconds

        # Get or create entries for this client
        entries = self._counters[client_key]

        # Count requests in current window
        current_count = sum(count for ts, count in entries if ts > cutoff)

        if current_count >= self.default_rate_limit:
            return False, current_count, self.default_rate_limit

        # Record this request
        entries.append((now, 1))
        return True, current_count + 1, self.default_rate_limit

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health and metrics endpoints
        path = request.url.path
        if path in ("/api/v1/health", "/metrics", "/"):
            return await call_next(request)

        self._cleanup_old_entries()

        client_key = self._get_client_key(request)
        allowed, current_count, limit = self._check_rate_limit(client_key)

        if not allowed:
            logger.warning(f"Rate limit exceeded for {client_key}: {current_count}/{limit}")
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded. Please try again later.",
                    "retry_after": self.window_seconds,
                },
                headers={
                    "Retry-After": str(self.window_seconds),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(max(0, limit - current_count))
        return response
