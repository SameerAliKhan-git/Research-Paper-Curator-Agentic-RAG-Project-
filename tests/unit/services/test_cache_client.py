"""Tests for the CacheClient (Redis exact-match + stale cache)."""

import json
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import fakeredis.aioredis
import pytest

from src.schemas.api.ask import AskRequest, AskResponse
from src.services.cache.client import CacheClient, STALE_TTL_MULTIPLIER


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_request(**overrides) -> AskRequest:
    defaults = {
        "query": "What are transformers?",
        "top_k": 3,
        "use_hybrid": True,
        "model": "llama3.2:1b",
        "categories": ["cs.AI"],
    }
    defaults.update(overrides)
    return AskRequest(**defaults)


def _make_response(**overrides) -> AskResponse:
    defaults = {
        "query": "What are transformers?",
        "answer": "Transformers are a neural architecture.",
        "sources": ["https://arxiv.org/pdf/1706.03762.pdf"],
        "chunks_used": 3,
        "search_mode": "hybrid",
    }
    defaults.update(overrides)
    return AskResponse(**defaults)


def _make_settings(ttl_hours: int = 6):
    """Return a minimal object that satisfies CacheClient.__init__."""
    return SimpleNamespace(ttl_hours=ttl_hours)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
async def redis():
    r = fakeredis.aioredis.FakeRedis()
    yield r
    await r.aclose()


@pytest.fixture
def settings():
    return _make_settings(ttl_hours=6)


@pytest.fixture
def cache_client(redis, settings):
    return CacheClient(redis_client=redis, settings=settings)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStoreAndRetrieveExactMatch:
    async def test_store_and_retrieve_exact_match(self, cache_client: CacheClient):
        req = _make_request()
        resp = _make_response()

        stored = await cache_client.store_response(req, resp)
        assert stored is True

        cached = await cache_client.find_cached_response(req)
        assert cached is not None
        assert cached.answer == resp.answer
        assert cached.sources == resp.sources
        assert cached.chunks_used == resp.chunks_used

    async def test_different_request_returns_none(self, cache_client: CacheClient):
        req1 = _make_request(query="What are CNNs?")
        req2 = _make_request(query="What are GANs?")
        resp = _make_response()

        await cache_client.store_response(req1, resp)
        cached = await cache_client.find_cached_response(req2)
        assert cached is None


class TestStaleCacheHasLongerTTL:
    async def test_stale_cache_has_longer_ttl(self, cache_client: CacheClient, redis):
        req = _make_request()
        resp = _make_response()

        await cache_client.store_response(req, resp)

        primary_key = cache_client._generate_cache_key(req)
        stale_key = cache_client._stale_key(req)

        primary_ttl = await redis.ttl(primary_key)
        stale_ttl = await redis.ttl(stale_key)

        # Both should have a positive TTL
        assert primary_ttl > 0
        assert stale_ttl > 0

        # Stale TTL should be STALE_TTL_MULTIPLIER times the primary TTL (approximately)
        assert stale_ttl > primary_ttl
        # Allow some slack due to timing between SET calls
        assert stale_ttl <= primary_ttl * STALE_TTL_MULTIPLIER + 2


class TestFindStaleReturnsWhenPrimaryExpired:
    async def test_find_stale_returns_when_primary_expired(self, cache_client: CacheClient, redis):
        req = _make_request()
        resp = _make_response()

        await cache_client.store_response(req, resp)

        primary_key = cache_client._generate_cache_key(req)

        # Expire the primary key immediately
        await redis.expire(primary_key, 0)
        # Force the deletion (expire(0) marks for expiry, delete is instant)
        await redis.delete(primary_key)

        # Primary cache should be empty
        assert await cache_client.find_cached_response(req) is None

        # Stale cache should still return the response
        stale = await cache_client.find_stale_response(req)
        assert stale is not None
        assert stale.answer == resp.answer
