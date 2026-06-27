import logging
from typing import Optional

import redis.asyncio as aioredis
from src.config import Settings
from src.services.auth.api_key_service import APIKeyService

logger = logging.getLogger(__name__)


async def make_api_key_service(settings: Settings) -> Optional[APIKeyService]:
    """Create API key service backed by async Redis.

    Returns None if Redis is unreachable. When None, auth middleware
    falls back to permissive mode (all requests allowed).
    """
    try:
        redis_settings = settings.redis
        client = aioredis.Redis(
            host=redis_settings.host,
            port=redis_settings.port,
            password=redis_settings.password if redis_settings.password else None,
            db=redis_settings.db,
            decode_responses=True,
            socket_timeout=redis_settings.socket_timeout,
            socket_connect_timeout=redis_settings.socket_connect_timeout,
            retry_on_timeout=True,
        )
        await client.ping()
        service = APIKeyService(client)
        logger.info("API key service initialized (async Redis)")
        return service
    except Exception as e:
        logger.warning(f"API key service unavailable (auth disabled): {e}")
        return None
