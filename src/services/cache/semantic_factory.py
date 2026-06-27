import logging
from typing import Optional

import redis.asyncio as aioredis
from src.config import Settings
from src.services.cache.factory import _get_connection_pool
from src.services.cache.semantic_cache import SemanticCache

logger = logging.getLogger(__name__)


async def make_semantic_cache(settings: Settings) -> Optional[SemanticCache]:
    """Create semantic cache client backed by the same Redis instance as exact cache.

    Returns None if Redis is unreachable or caching is disabled. The caller
    should treat None as "caching unavailable" and proceed without it.
    """
    try:
        pool = _get_connection_pool(settings)
        client = aioredis.Redis(connection_pool=pool)
        await client.ping()
        cache = SemanticCache(client, settings.redis)
        logger.info("Semantic cache initialized (shared connection pool)")
        return cache
    except Exception as e:
        logger.warning(f"Semantic cache unavailable (disabled): {e}")
        return None
