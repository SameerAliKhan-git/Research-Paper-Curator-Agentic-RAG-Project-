import hashlib
import json
import logging
import time
from typing import Optional

import redis.asyncio as aioredis
from fastapi import Depends, HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from src.config import Settings

logger = logging.getLogger(__name__)

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

AUTH_PREFIX = "auth:keys"
QUOTA_PREFIX = "auth:quota"
RATE_PREFIX = "auth:rate"

# Lua script for atomic rate-limit INCR+EXPIRE (prevents TOCTOU between INCR and EXPIRE)
# Returns: current count after increment, or -2 if exhausted
_DECR_QUOTA_LUA = """
local key = KEYS[1]
local quota_key = KEYS[2]
local current = tonumber(redis.call('GET', quota_key) or '-1')
if current == -1 then
    -- First call: initialize from metadata quota
    local meta_raw = redis.call('GET', key)
    if not meta_raw then
        return -1
    end
    local meta = cjson.decode(meta_raw)
    local quota = tonumber(meta['quota_remaining'] or -1)
    if quota <= 0 then
        return -2
    end
    local new_val = quota - 1
    redis.call('SET', quota_key, new_val, 'EX', 86400)
    return new_val
elseif current <= 0 then
    return -2
else
    local new_val = redis.call('DECR', quota_key)
    if new_val < 0 then
        redis.call('SET', quota_key, 0, 'EX', 86400)
        return -2
    end
    return new_val
end
"""

# Lua script for atomic rate-limit window increment with TTL
# KEYS[1] = rate limit window key
# ARGV[1] = rate limit count
# ARGV[2] = window TTL in seconds
# Returns: 1 if allowed, 0 if rate limited
_RATE_LIMIT_LUA = """
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local ttl = tonumber(ARGV[2])
local current = redis.call('INCR', key)
if current == 1 then
    redis.call('EXPIRE', key, ttl)
end
if current > limit then
    return 0
end
return 1
"""


class APIKeyMetadata:
    """Represents a validated API key and its associated metadata."""

    __slots__ = ("key_hash", "user_id", "tier", "rate_limit", "quota_remaining", "tenants")

    def __init__(
        self,
        key_hash: str,
        user_id: str,
        tier: str = "standard",
        rate_limit: int = 60,
        quota_remaining: int = -1,
        tenants: Optional[list[str]] = None,
    ):
        self.key_hash = key_hash
        self.user_id = user_id
        self.tier = tier
        self.rate_limit = rate_limit
        self.quota_remaining = quota_remaining
        self.tenants = tenants


def _hash_api_key(raw_key: str) -> str:
    """SHA-256 hash of the raw API key for storage."""
    return hashlib.sha256(raw_key.encode()).hexdigest()


class APIKeyService:
    """Manages API key validation, rate limiting, and quota tracking.

    Keys are stored in Redis as a hash: ``auth:keys:{sha256}`` -> JSON metadata.
    Rate limiting uses a sliding-window counter per key per minute.

    Key lifecycle:
        1. Admin creates key via ``create_key()``
        2. Client sends key in ``X-API-Key`` header
        3. Middleware validates, checks rate limit, decrements quota
        4. On quota exhaustion or rate limit breach -> 429 / 403
    """

    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self._decr_quota_script = self.redis.register_script(_DECR_QUOTA_LUA)
        self._rate_limit_script = self.redis.register_script(_RATE_LIMIT_LUA)

    async def create_key(
        self,
        raw_key: str,
        user_id: str,
        tier: str = "standard",
        rate_limit: int = 60,
        daily_quota: int = 1000,
        tenants: Optional[list[str]] = None,
    ) -> dict:
        """Register a new API key.

        Returns the key metadata dict (including the hash, not the raw key).
        The raw key should be shown to the user once and never stored.
        """
        key_hash = _hash_api_key(raw_key)
        metadata = {
            "user_id": user_id,
            "tier": tier,
            "rate_limit": rate_limit,
            "daily_quota": daily_quota,
            "quota_remaining": daily_quota,
            "created_at": int(time.time()),
            "enabled": True,
            "tenants": tenants,
        }
        await self.redis.set(f"{AUTH_PREFIX}:{key_hash}", json.dumps(metadata))
        logger.info(f"Created API key for user={user_id}, tier={tier}")
        return {"key_hash": key_hash, **metadata}

    async def validate_key(self, raw_key: str) -> Optional[APIKeyMetadata]:
        """Validate an API key and return its metadata.

        Returns None if the key is invalid or disabled.
        """
        key_hash = _hash_api_key(raw_key)
        raw = await self.redis.get(f"{AUTH_PREFIX}:{key_hash}")

        if raw is None:
            return None

        meta = json.loads(raw)
        if not meta.get("enabled", False):
            return None

        return APIKeyMetadata(
            key_hash=key_hash,
            user_id=meta["user_id"],
            tier=meta.get("tier", "standard"),
            rate_limit=meta.get("rate_limit", 60),
            quota_remaining=meta.get("quota_remaining", -1),
            tenants=meta.get("tenants"),
        )

    async def check_rate_limit(self, key_hash: str, rate_limit: int) -> bool:
        """Sliding-window rate limiter: max ``rate_limit`` requests per 60-second window.

        Returns True if the request is allowed, False if rate limited.
        Uses a Lua script for atomic INCR+EXPIRE (no TOCTOU gap).
        """
        window_key = f"{RATE_PREFIX}:{key_hash}:{int(time.time() // 60)}"
        allowed = await self._rate_limit_script(
            keys=[window_key],
            args=[rate_limit, 120],
        )
        if isinstance(allowed, bytes):
            allowed = int(allowed)
        return allowed == 1

    async def decrement_quota(self, key_hash: str) -> int:
        """Decrement daily quota atomically using Lua script. Returns remaining quota (-1 if unlimited, -2 if exhausted)."""
        meta_raw = await self.redis.get(f"{AUTH_PREFIX}:{key_hash}")
        if meta_raw is None:
            return -1

        meta = json.loads(meta_raw)
        if meta.get("quota_remaining", -1) == -1:
            return -1

        quota_key = f"{QUOTA_PREFIX}:{key_hash}"
        auth_key = f"{AUTH_PREFIX}:{key_hash}"

        remaining = await self._decr_quota_script(
            keys=[auth_key, quota_key],
        )

        if isinstance(remaining, bytes):
            remaining = int(remaining)

        # Do not write back to static key metadata in Redis, avoiding concurrent write race conditions.
        # Dynamic quota is tracked atomically in quota_key.
        return remaining

    async def revoke_key(self, raw_key: str) -> bool:
        """Disable an API key without deleting it."""
        key_hash = _hash_api_key(raw_key)
        meta_raw = await self.redis.get(f"{AUTH_PREFIX}:{key_hash}")
        if meta_raw is None:
            return False

        meta = json.loads(meta_raw)
        meta["enabled"] = False
        await self.redis.set(f"{AUTH_PREFIX}:{key_hash}", json.dumps(meta))
        logger.info(f"Revoked API key user={meta.get('user_id')}")
        return True


def get_api_key_service(request: Request) -> APIKeyService:
    """FastAPI dependency that retrieves the APIKeyService from app state."""
    return getattr(request.app.state, "api_key_service", None)


async def require_api_key(
    request: Request,
    x_api_key: Optional[str] = Security(API_KEY_HEADER),
) -> APIKeyMetadata:
    """FastAPI dependency that enforces API key authentication.

    If the X-API-Key header is missing or invalid, returns 401/403.
    On success, returns the validated APIKeyMetadata for downstream use.

    Usage in routers::

        @router.get("/protected")
        async def protected_endpoint(
            key: APIKeyMetadata = Depends(require_api_key),
        ):
            return {"user": key.user_id}
    """
    service: Optional[APIKeyService] = getattr(request.app.state, "api_key_service", None)

    # If auth service is not initialized (e.g. no Redis), fail closed — deny access
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Authentication service unavailable. Please try again later.",
        )

    if x_api_key is None:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")

    metadata = await service.validate_key(x_api_key)
    if metadata is None:
        raise HTTPException(status_code=403, detail="Invalid or revoked API key")

    # Rate limit check
    if not await service.check_rate_limit(metadata.key_hash, metadata.rate_limit):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")

    # Quota check + atomic decrement (single Lua script, no TOCTOU)
    remaining = await service.decrement_quota(metadata.key_hash)
    if remaining == -2:
        raise HTTPException(status_code=429, detail="Daily quota exhausted. Contact support.")

    # Store on request.state for downstream middleware/logging
    request.state.api_key_metadata = metadata

    # Tenant authorization check (API Key allowed tenants vs request tenant)
    tenant_ctx = getattr(request.state, "tenant", None)
    if tenant_ctx is not None and metadata.tenants is not None:
        if tenant_ctx.tenant_id not in metadata.tenants:
            raise HTTPException(
                status_code=403,
                detail=f"Tenant '{tenant_ctx.tenant_id}' is not authorized for this API key.",
            )

    return metadata
