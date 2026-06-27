"""OpenTelemetry tracing setup for the FastAPI RAG application."""

import logging
from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


def setup_opentelemetry(
    service_name: str,
    environment: str,
    otlp_endpoint: Optional[str] = None,
) -> TracerProvider:
    """Configure OpenTelemetry tracing with optional OTLP export.

    Args:
        service_name: Name of the service for trace identification.
        environment: Deployment environment (e.g. "development", "production").
        otlp_endpoint: Optional OTLP collector endpoint URL.

    Returns:
        Configured TracerProvider instance.
    """
    resource = Resource.create(
        {
            SERVICE_NAME: service_name,
            "deployment.environment": environment,
            SERVICE_VERSION: "0.1.0",
        }
    )

    provider = TracerProvider(resource=resource)

    if otlp_endpoint:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )

        exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
        logger.info("Tracing exporting to OTLP endpoint: %s", otlp_endpoint)
    else:
        exporter = ConsoleSpanExporter()
        logger.info("Tracing exporting to console (development mode)")

    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    logger.info(
        "OpenTelemetry configured: service=%s, environment=%s",
        service_name,
        environment,
    )
    return provider


def get_tracer(name: str) -> trace.Tracer:
    """Get a tracer from the global TracerProvider.

    Args:
        name: Name of the tracer (typically __name__ of the calling module).

    Returns:
        A Tracer instance from the global provider.
    """
    return trace.get_tracer(name)


class TracingMiddleware(BaseHTTPMiddleware):
    """Middleware that creates spans for each HTTP request.

    Records request metadata as span attributes and captures exceptions.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        tracer = trace.get_tracer("fastapi")

        with tracer.start_as_current_span(f"{request.method} {request.url.path}") as span:
            span.set_attribute("http.method", request.method)
            span.set_attribute("http.url", str(request.url))
            span.set_attribute("http.target", request.url.path)
            span.set_attribute("http.scheme", request.url.scheme)
            span.set_attribute(
                "http.request_id",
                getattr(request.state, "correlation_id", "unknown"),
            )

            try:
                response = await call_next(request)
                span.set_attribute("http.status_code", response.status_code)
                span.set_attribute(
                    "http.response.size",
                    int(response.headers.get("content-length", 0)),
                )
                return response
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(trace.StatusCode.ERROR, str(exc))
                raise
