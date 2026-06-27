"""Tests for RequestLoggingMiddleware, ExceptionCaptureMiddleware, and RequestBodySizeLimitMiddleware."""

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field

from src.middlewares import (
    ExceptionCaptureMiddleware,
    RequestBodySizeLimitMiddleware,
    RequestLoggingMiddleware,
)


# ---------------------------------------------------------------------------
# Minimal test app – avoids importing the full application with all its
# heavy dependencies (Postgres, OpenSearch, Ollama, …).
# ---------------------------------------------------------------------------

class DummyBody(BaseModel):
    value: int = Field(...)


_test_app = FastAPI(title="middleware-test")

_test_app.add_middleware(RequestLoggingMiddleware)
_test_app.add_middleware(ExceptionCaptureMiddleware)
_test_app.add_middleware(RequestBodySizeLimitMiddleware, max_size_bytes=1024)  # 1 KB


@_test_app.get("/ping")
async def ping():
    return {"status": "ok"}


@_test_app.post("/validate")
async def validate(body: DummyBody):
    return {"echo": body.value}


@_test_app.get("/explode")
async def explode():
    raise RuntimeError("boom")


@pytest.fixture()
def app():
    """Yield the minimal test app (reset state if needed)."""
    yield _test_app


# ---------------------------------------------------------------------------
# Tests – RequestLoggingMiddleware
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_request_logging_adds_correlation_id(app: FastAPI):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/ping")

    assert resp.status_code == 200
    assert "X-Correlation-ID" in resp.headers
    # Correlation ID is first 8 chars of a uuid4 hex
    cid = resp.headers["X-Correlation-ID"]
    assert len(cid) == 8
    assert all(c in "0123456789abcdef" for c in cid)


@pytest.mark.anyio
async def test_request_logging_adds_response_time(app: FastAPI):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/ping")

    assert "X-Response-Time" in resp.headers
    rt = resp.headers["X-Response-Time"]
    assert rt.endswith("ms")
    # The numeric part should be parseable as a float
    float(rt.rstrip("ms"))


# ---------------------------------------------------------------------------
# Tests – ExceptionCaptureMiddleware
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_exception_capture_returns_500_json(app: FastAPI):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/explode")

    assert resp.status_code == 500
    body = resp.json()
    assert body == {"detail": "Internal server error"}


@pytest.mark.anyio
async def test_exception_capture_allows_validation_error(app: FastAPI):
    """RequestValidationError (422) should still be raised normally."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # POST /validate expects a body with an int field – send empty
        resp = await client.post("/validate", json={})

    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Tests – RequestBodySizeLimitMiddleware
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_body_size_limit_allows_small_request(app: FastAPI):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/ping")

    assert resp.status_code == 200


@pytest.mark.anyio
async def test_body_size_limit_rejects_large_request(app: FastAPI):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Payload exceeds 1 KB limit set in the app
        payload = "x" * 2048
        resp = await client.post(
            "/validate",
            content=payload,
            headers={"Content-Type": "application/json", "Content-Length": str(len(payload))},
        )

    assert resp.status_code == 413
    body = resp.json()
    assert "too large" in body["detail"].lower()


@pytest.mark.anyio
async def test_body_size_limit_ignores_missing_content_length(app: FastAPI):
    """Requests without a Content-Length header (e.g. streaming) pass through."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/ping")

    assert resp.status_code == 200
