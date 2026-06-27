import hashlib
import json
import logging
from datetime import timedelta
from typing import Optional

import redis.asyncio as aioredis
from src.config import RedisSettings
from src.schemas.api.ask import AskRequest, AskResponse

logger = logging.getLogger(__name__)

# Stale cache extends TTL by this factor for graceful degradation
STALE_TTL_MULTIPLIER = 4


class CacheClient:
    """Redis-based exact match cache for RAG queries.

    Supports two TTL windows:
    - Primary TTL (configurable, default 6h): normal cache entries
    - Stale TTL (4x primary): entries kept for graceful degradation when upstream services fail
    """

    def __init__(self, redis_client: aioredis.Redis, settings: RedisSettings):
        self.redis = redis_client
        self.settings = settings
        self.ttl = timedelta(hours=settings.ttl_hours)
        self.stale_ttl = self.ttl * STALE_TTL_MULTIPLIER

    def _generate_cache_key(self, request: AskRequest, tenant_id: Optional[str] = None) -> str:
        """Generate exact cache key based on request parameters."""
        key_data = {
            "query": request.query,
            "model": request.model,
            "top_k": request.top_k,
            "use_hybrid": request.use_hybrid,
            "categories": sorted(request.categories) if request.categories else [],
        }
        key_string = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:16]
        prefix = f"tenant:{tenant_id}:" if tenant_id else ""
        return f"{prefix}exact_cache:{key_hash}"

    def _stale_key(self, request: AskRequest, tenant_id: Optional[str] = None) -> str:
        """Generate stale cache key (separate from primary)."""
        key_data = {
            "query": request.query,
            "model": request.model,
            "top_k": request.top_k,
            "use_hybrid": request.use_hybrid,
            "categories": sorted(request.categories) if request.categories else [],
        }
        key_string = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:16]
        prefix = f"tenant:{tenant_id}:" if tenant_id else ""
        return f"{prefix}exact_cache:stale:{key_hash}"

    async def find_cached_response(self, request: AskRequest, tenant_id: Optional[str] = None) -> Optional[AskResponse]:
        """Find cached response for exact query match."""
        try:
            cache_key = self._generate_cache_key(request, tenant_id=tenant_id)

            # Simple Redis GET operation - O(1)
            cached_response = await self.redis.get(cache_key)

            if cached_response:
                try:
                    response_data = json.loads(cached_response)
                    logger.info("Cache hit for exact query match")
                    return AskResponse(**response_data)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to deserialize cached response: {e}")
                    return None

            return None

        except Exception as e:
            logger.error(f"Error checking cache: {e}")
            return None

    async def find_stale_response(self, request: AskRequest, tenant_id: Optional[str] = None) -> Optional[AskResponse]:
        """Find stale cached response for graceful degradation.

        Used when upstream services (Ollama, OpenSearch, etc.) are unavailable.
        Returns a response that may be slightly outdated but better than an error.
        """
        try:
            stale_key = self._stale_key(request, tenant_id=tenant_id)
            cached_response = await self.redis.get(stale_key)

            if cached_response:
                try:
                    response_data = json.loads(cached_response)
                    logger.info("Stale cache hit for graceful degradation")
                    return AskResponse(**response_data)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to deserialize stale cached response: {e}")
                    return None

            return None

        except Exception as e:
            logger.error(f"Error checking stale cache: {e}")
            return None

    async def store_response(self, request: AskRequest, response: AskResponse, tenant_id: Optional[str] = None) -> bool:
        """Store response for exact query matching.

        Also stores in the stale cache with a longer TTL for graceful degradation.
        """
        try:
            cache_key = self._generate_cache_key(request, tenant_id=tenant_id)

            # Simple Redis SET operation with TTL
            success = await self.redis.set(cache_key, response.model_dump_json(), ex=self.ttl)

            # Also store in stale cache with extended TTL
            stale_key = self._stale_key(request, tenant_id=tenant_id)
            await self.redis.set(stale_key, response.model_dump_json(), ex=self.stale_ttl)

            if success:
                logger.info(f"Stored response in exact cache with key {cache_key[:16]}...")
                return True
            else:
                logger.warning("Failed to store response in cache")
                return False

        except Exception as e:
            logger.error(f"Error storing in cache: {e}")
            return False

    async def clear(self) -> bool:
        """Clear all exact-match cached queries across all tenants."""
        try:
            cursor = 0
            keys_deleted = 0
            while True:
                cursor, keys = await self.redis.scan(cursor, match="*exact_cache:*", count=100)
                if keys:
                    await self.redis.delete(*keys)
                    keys_deleted += len(keys)
                if cursor == 0:
                    break
            logger.info(f"Cleared exact-match cache: deleted {keys_deleted} keys")
            return True
        except Exception as e:
            logger.error(f"Error clearing exact-match cache: {e}")
            return False
