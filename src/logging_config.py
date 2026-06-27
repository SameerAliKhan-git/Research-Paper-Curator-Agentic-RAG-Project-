import logging
import sys
from typing import Any, MutableMapping

import structlog
from starlette.types import ASGIApp, Receive, Scope, Send


def _add_log_level(logger: Any, method_name: str, event_dict: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    if method_name and "log_level" not in event_dict:
        event_dict["log_level"] = method_name.upper()
    return event_dict


def _add_timestamp(logger: Any, method_name: str, event_dict: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    if "timestamp" not in event_dict:
        from datetime import datetime, timezone

        event_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
    return event_dict


def _add_logger_name(logger: Any, method_name: str, event_dict: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    if "logger" not in event_dict:
        # structlog BoundLogger exposes .name
        if hasattr(logger, "name"):
            event_dict["logger"] = logger.name
    return event_dict


def _json_renderer(logger: Any, method_name: str, event_dict: MutableMapping[str, Any]) -> str:
    import json

    return json.dumps(event_dict, default=str)


def _twisted_style_renderer(logger: Any, method_name: str, event_dict: MutableMapping[str, Any]) -> str:
    level = event_dict.pop("log_level", method_name.upper())
    logger_name = event_dict.pop("logger", "")
    timestamp = event_dict.pop("timestamp", "")
    message = event_dict.pop("event", "")
    extras = " ".join(f"{k}={v!r}" for k, v in event_dict.items())
    parts = [f"[{timestamp}]" if timestamp else "", f"[{level}]", f"[{logger_name}]" if logger_name else "", message]
    if extras:
        parts.append(extras)
    return " ".join(part for part in parts if part)


def setup_logging(environment: str = "development") -> None:
    """Configure structlog with environment-aware rendering.

    Args:
        environment: One of "development", "staging", "production".
    """
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        _add_log_level,
        _add_timestamp,
        _add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if environment == "production":
        renderer: structlog.types.Processor = _json_renderer
        wrapper_class = structlog.make_filtering_bound_logger(logging.INFO)
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)
        wrapper_class = structlog.make_filtering_bound_logger(logging.DEBUG)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging to output structlog in production
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

    # Silence noisy third-party loggers
    for noisy in ("uvicorn.access", "httpcore", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class LogContextMiddleware:
    """ASGI middleware that binds correlation_id to structlog context for each request."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> Any:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        structlog.contextvars.clear_contextvars()

        correlation_id = ""
        state = scope.get("state")
        if state is not None:
            # state is a starlette.datastructures.State object, not a dict
            correlation_id = getattr(state, "correlation_id", "") or ""

        if correlation_id:
            structlog.contextvars.bind_contextvars(correlation_id=correlation_id)

        try:
            return await self.app(scope, receive, send)
        finally:
            structlog.contextvars.clear_contextvars()


def log_context_middleware(app: ASGIApp) -> LogContextMiddleware:
    """Factory that returns a LogContextMiddleware wrapping the given ASGI app."""
    return LogContextMiddleware(app)
