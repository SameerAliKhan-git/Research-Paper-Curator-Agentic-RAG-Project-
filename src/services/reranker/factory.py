"""Factory for creating reranker service instances."""

import asyncio
import logging
import threading
from typing import Optional

from src.config import get_settings

from .client import RerankerClient, create_reranker_client

logger = logging.getLogger(__name__)

_reranker_client: Optional[RerankerClient] = None
_reranker_lock = threading.Lock()


def make_reranker_service() -> Optional[RerankerClient]:
    """Create or return cached reranker client instance (thread-safe singleton).

    Returns:
        RerankerClient instance or None if disabled
    """
    global _reranker_client

    if _reranker_client is not None:
        return _reranker_client

    with _reranker_lock:
        if _reranker_client is not None:
            return _reranker_client

        settings = get_settings()

        if not settings.reranker.enabled:
            logger.info("Reranker disabled in settings")
            return None

        try:
            _reranker_client = create_reranker_client(settings.reranker)
            logger.info(f"Reranker service initialized: {settings.reranker.provider}/{settings.reranker.model}")
            return _reranker_client
        except Exception as e:
            logger.warning(f"Failed to initialize reranker: {e}")
            return None


async def reset_reranker_service():
    """Reset the cached reranker client (async-safe)."""
    global _reranker_client
    with _reranker_lock:
        if _reranker_client is not None:
            try:
                await _reranker_client.close()
            except Exception:
                pass
        _reranker_client = None
