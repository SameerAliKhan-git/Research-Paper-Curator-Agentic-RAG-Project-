"""Prometheus metrics collection and exposition for the RAG API."""

import re
import time

import prometheus_client
from fastapi import Request, Response
from fastapi.responses import Response as FastAPIResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

_EXCLUDED_PATHS = frozenset({"/health", "/metrics"})
_PATH_PARAM_RE = re.compile(r"/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|/\d+")


def _sanitize_endpoint(path: str) -> str:
    if path.startswith("/api/v1"):
        return "/api/v1/..."
    return _PATH_PARAM_RE.sub("/_id", path)


class MetricsCollector:
    def __init__(self) -> None:
        self.http_requests_total = prometheus_client.Counter(
            "http_requests_total",
            "Total HTTP requests",
            labelnames=["method", "endpoint", "status_code"],
        )
        self.http_request_duration_seconds = prometheus_client.Histogram(
            "http_request_duration_seconds",
            "HTTP request latency in seconds",
            labelnames=["method", "endpoint"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
        )
        self.llm_requests_total = prometheus_client.Counter(
            "llm_requests_total",
            "Total LLM requests",
            labelnames=["model", "status"],
        )
        self.llm_duration_seconds = prometheus_client.Histogram(
            "llm_duration_seconds",
            "LLM request latency in seconds",
            labelnames=["model"],
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
        )
        self.llm_tokens_total = prometheus_client.Counter(
            "llm_tokens_total",
            "Total LLM tokens consumed",
            labelnames=["model", "type"],
        )
        self.cache_hits_total = prometheus_client.Counter(
            "cache_hits_total",
            "Total cache hits",
            labelnames=["cache_type"],
        )
        self.cache_misses_total = prometheus_client.Counter(
            "cache_misses_total",
            "Total cache misses",
            labelnames=["cache_type"],
        )
        self.search_requests_total = prometheus_client.Counter(
            "search_requests_total",
            "Total search requests",
            labelnames=["mode"],
        )
        self.search_duration_seconds = prometheus_client.Histogram(
            "search_duration_seconds",
            "Search request latency in seconds",
            labelnames=["mode"],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
        )
        # P95/P99 latency histograms for critical operations
        self.rag_latency_seconds = prometheus_client.Histogram(
            "rag_latency_seconds",
            "End-to-end RAG request latency in seconds",
            labelnames=["endpoint"],
            buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0),
        )
        self.generation_latency_seconds = prometheus_client.Histogram(
            "generation_latency_seconds",
            "LLM generation latency in seconds",
            labelnames=["model"],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0),
        )
        self.retrieval_latency_seconds = prometheus_client.Histogram(
            "retrieval_latency_seconds",
            "Document retrieval latency in seconds",
            labelnames=["mode"],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
        )
        self.reranker_latency_seconds = prometheus_client.Histogram(
            "reranker_latency_seconds",
            "Reranker latency in seconds",
            labelnames=["provider"],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
        )
        # Cache hit rate gauge (computed from counters)
        self.cache_hit_rate = prometheus_client.Gauge(
            "cache_hit_rate",
            "Cache hit rate (hits / (hits + misses))",
            labelnames=["cache_type"],
        )
        self.guardrail_rejections_total = prometheus_client.Counter(
            "guardrail_rejections_total",
            "Total queries rejected by guardrail",
            labelnames=["reason"],
        )
        self.agentic_workflow_steps = prometheus_client.Histogram(
            "agentic_workflow_steps",
            "Number of steps in agentic workflow",
            buckets=(1, 2, 3, 4, 5, 6, 7, 8),
        )


class PrometheusMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, metrics_collector: MetricsCollector) -> None:
        super().__init__(app)
        self.metrics = metrics_collector

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path
        if path in _EXCLUDED_PATHS:
            return await call_next(request)

        method = request.method
        endpoint = _sanitize_endpoint(path)
        start = time.perf_counter()

        response = await call_next(request)

        elapsed = time.perf_counter() - start
        status_code = str(response.status_code)

        self.metrics.http_requests_total.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
        self.metrics.http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(elapsed)

        return response


def get_metrics_endpoint():
    async def _metrics() -> Response:
        return FastAPIResponse(
            content=prometheus_client.generate_latest(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    return _metrics


metrics = MetricsCollector()
