import logging
from typing import Optional

import redis.asyncio as aioredis
from src.config import Settings
from src.services.cache.client import CacheClient

logger = logging.getLogger(__name__)

# Shared connection pool for all Redis clients
_redis_connection_pool: Optional[aioredis.ConnectionPool] = None


def _get_connection_pool(settings: Settings) -> aioredis.ConnectionPool:
    """Get or create a shared Redis connection pool.

    Connection pooling is critical for production workloads to avoid
    creating new connections for every request.
    """
    global _redis_connection_pool
    if _redis_connection_pool is not None:
        return _redis_connection_pool

    redis_settings = settings.redis
    _redis_connection_pool = aioredis.ConnectionPool(
        host=redis_settings.host,
        port=redis_settings.port,
        password=redis_settings.password if redis_settings.password else None,
        db=redis_settings.db,
        decode_responses=redis_settings.decode_responses,
        socket_timeout=redis_settings.socket_timeout,
        socket_connect_timeout=redis_settings.socket_connect_timeout,
        retry_on_timeout=True,
        max_connections=20,
    )
    return _redis_connection_pool


async def make_async_redis_client(settings: Settings) -> aioredis.Redis:
    """Create async Redis client with connection pooling."""
    try:
        pool = _get_connection_pool(settings)
        client = aioredis.Redis(connection_pool=pool)

        await client.ping()
        logger.info(f"Connected to Redis at {settings.redis.host}:{settings.redis.port} (pool max={pool.max_connections})")
        return client

    except aioredis.ConnectionError as e:
        logger.error(f"Failed to connect to Redis: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating Redis client: {e}")
        raise


async def make_cache_client(settings: Settings) -> Optional[CacheClient]:
    """Create exact match cache client."""
    try:
        redis_client = await make_async_redis_client(settings)
        cache_client = CacheClient(redis_client, settings.redis)
        logger.info("Exact match cache client created successfully")
        return cache_client
    except Exception as e:
        logger.warning(f"Failed to create cache client (caching disabled): {e}")
        return None
