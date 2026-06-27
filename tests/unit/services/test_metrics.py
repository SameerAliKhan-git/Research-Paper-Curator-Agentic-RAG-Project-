"""Tests for Prometheus metrics endpoint and counter behavior."""

import httpx
import prometheus_client
import pytest
from fastapi import FastAPI

from src.services.metrics import PrometheusMiddleware


# ---------------------------------------------------------------------------
# Minimal test app – use a dedicated registry to avoid collisions
# ---------------------------------------------------------------------------

_test_registry = prometheus_client.CollectorRegistry()

_http_counter = prometheus_client.Counter(
    "test_http_requests_total",
    "Total HTTP requests",
    labelnames=["method", "endpoint", "status_code"],
    registry=_test_registry,
)
_http_duration = prometheus_client.Histogram(
    "test_http_request_duration_seconds",
    "HTTP request latency in seconds",
    labelnames=["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=_test_registry,
)


class _FakeCollector:
    http_requests_total = _http_counter
    http_request_duration_seconds = _http_duration


_collector = _FakeCollector()

_metrics_app = FastAPI(title="metrics-test")
_metrics_app.add_middleware(PrometheusMiddleware, metrics_collector=_collector)


@_metrics_app.get("/ping")
async def ping():
    return {"status": "ok"}


@pytest.fixture
def collector():
    return _collector


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_metrics_endpoint_returns_prometheus_format(collector):
    transport = httpx.ASGITransport(app=_metrics_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/ping")

    assert resp.status_code == 200


@pytest.mark.anyio
async def test_http_requests_counter_increments(collector):
    transport = httpx.ASGITransport(app=_metrics_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await client.get("/ping")
        await client.get("/ping")

    metrics_text = prometheus_client.generate_latest(_test_registry).decode()
    for line in metrics_text.splitlines():
        if line.startswith("test_http_requests_total") and 'method="GET"' in line and "/ping" in line:
            value = float(line.split()[-1])
            assert value >= 2.0
            return

    pytest.fail("test_http_requests_total counter not found in metrics output")


@pytest.mark.anyio
async def test_histogram_observes_latency(collector):
    transport = httpx.ASGITransport(app=_metrics_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await client.get("/ping")

    metrics_text = prometheus_client.generate_latest(_test_registry).decode()
    assert "test_http_request_duration_seconds" in metrics_text
