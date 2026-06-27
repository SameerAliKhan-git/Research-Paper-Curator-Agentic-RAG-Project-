"""Tests for semantic cache service."""

import asyncio
import hashlib
import json
import numpy as np
import pytest
import fakeredis.aioredis
from unittest.mock import MagicMock

from src.schemas.api.ask import AskRequest, AskResponse
from src.services.cache.semantic_cache import SemanticCache


@pytest.fixture
async def mock_redis():
    """Fake Redis client."""
    r = fakeredis.aioredis.FakeRedis()
    yield r
    await r.aclose()


@pytest.fixture
def mock_settings():
    """Mock Redis settings."""
    settings = MagicMock()
    settings.ttl_hours = 6
    return settings


@pytest.fixture
def semantic_cache(mock_redis, mock_settings):
    """Create SemanticCache instance."""
    return SemanticCache(mock_redis, mock_settings)


class TestSemanticCache:
    """Tests for SemanticCache."""

    def test_serialize_vector(self):
        """Test vector serialization."""
        vector = [0.1, 0.2, 0.3]
        serialized = SemanticCache._serialize_vector(vector)
        assert isinstance(serialized, bytes)
        assert len(serialized) == len(vector) * 4  # 4 bytes per float

    def test_deserialize_vector(self):
        """Test vector deserialization."""
        original = [0.1, 0.2, 0.3]
        serialized = SemanticCache._serialize_vector(original)
        deserialized = SemanticCache._deserialize_vector(serialized)
        np.testing.assert_array_almost_equal(deserialized, original, decimal=6)

    def test_cosine_similarity_identical(self):
        """Test cosine similarity of identical vectors."""
        vec = np.array([1.0, 0.0, 0.0])
        score = SemanticCache._cosine_similarity(vec, vec)
        assert score == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity of orthogonal vectors."""
        vec_a = np.array([1.0, 0.0])
        vec_b = np.array([0.0, 1.0])
        score = SemanticCache._cosine_similarity(vec_a, vec_b)
        assert score == pytest.approx(0.0)

    def test_cosine_similarity_opposite(self):
        """Test cosine similarity of opposite vectors."""
        vec_a = np.array([1.0, 0.0])
        vec_b = np.array([-1.0, 0.0])
        score = SemanticCache._cosine_similarity(vec_a, vec_b)
        assert score == pytest.approx(-1.0)

    def test_exact_key_generation(self, semantic_cache):
        """Test exact cache key generation."""
        request = AskRequest(query="test query")
        key = semantic_cache._exact_key(request)
        assert key.startswith("exact_cache:")
        assert len(key) > 20  # Should have hash

    def test_exact_key_deterministic(self, semantic_cache):
        """Test that same request produces same key."""
        request = AskRequest(query="test query")
        key1 = semantic_cache._exact_key(request)
        key2 = semantic_cache._exact_key(request)
        assert key1 == key2

    async def test_find_exact_hit(self, semantic_cache, mock_redis):
        """Test exact cache hit."""
        request = AskRequest(query="test query")
        cached_response = AskResponse(
            query="test query",
            answer="cached answer",
            sources=[],
            chunks_used=3,
            search_mode="hybrid",
        )

        key = semantic_cache._exact_key(request)
        await mock_redis.set(key, cached_response.model_dump_json())

        result = await semantic_cache.find_exact(request)
        assert result is not None
        assert result.answer == "cached answer"

    async def test_find_exact_miss(self, semantic_cache, mock_redis):
        """Test exact cache miss."""
        request = AskRequest(query="test query")
        result = await semantic_cache.find_exact(request)
        assert result is None

    async def test_store_exact_cache(self, semantic_cache, mock_redis):
        """Test storing response in exact cache."""
        request = AskRequest(query="test query")
        response = AskResponse(
            query="test query",
            answer="test answer",
            sources=[],
            chunks_used=1,
            search_mode="bm25",
        )
        query_embedding = [0.1] * 1024

        result = await semantic_cache.store(request, response, query_embedding)
        assert result is True

        # Check exact key got populated in Redis
        key = semantic_cache._exact_key(request)
        val = await mock_redis.get(key)
        assert val is not None

    async def test_find_semantic_hit(self, semantic_cache, mock_redis):
        """Test semantic cache hit."""
        request = AskRequest(query="test query")
        response = AskResponse(
            query="test query",
            answer="semantic answer",
            sources=[],
            chunks_used=1,
            search_mode="bm25",
        )
        embedding = [0.1] * 1024

        # Store first
        await semantic_cache.store(request, response, embedding)

        # Now lookup with identical embedding
        result = await semantic_cache.find_semantic(embedding, request)
        assert result is not None
        assert result.answer == "semantic answer"

    async def test_cleanup_orphans(self, semantic_cache, mock_redis):
        """Test cleaning up orphaned semantic entries."""
        request = AskRequest(query="test query")
        response = AskResponse(
            query="test query",
            answer="answer",
            sources=[],
            chunks_used=1,
            search_mode="bm25",
        )
        embedding = [0.1] * 1024

        # Store
        await semantic_cache.store(request, response, embedding)

        # Retrieve vectors hash from the index set
        index_key = semantic_cache._index_key()
        hashes = await mock_redis.smembers(index_key)
        assert len(hashes) == 1

        vhash = list(hashes)[0].decode()
        vec_key = semantic_cache._embedding_key(vhash)

        # Delete the vector key to simulate TTL expiration of the vector key but index set still holding the hash
        await mock_redis.delete(vec_key)

        # Cleanup orphans
        removed = await semantic_cache.cleanup_orphans()
        assert removed == 1

        # Check index is now empty
        hashes_after = await mock_redis.smembers(index_key)
        assert len(hashes_after) == 0

    async def test_invalidate_similar_hit(self, semantic_cache, mock_redis):
        """Test selective invalidation when similarity is above threshold."""
        request = AskRequest(query="test query")
        response = AskResponse(
            query="test query",
            answer="semantic answer",
            sources=[],
            chunks_used=1,
            search_mode="bm25",
        )
        embedding = [0.1] * 1024

        # Store
        await semantic_cache.store(request, response, embedding)

        # Confirm stored
        exact_key = semantic_cache._exact_key(request)
        assert await mock_redis.get(exact_key) is not None

        # Invalidate with similar embedding (similarity 1.0)
        invalidated = await semantic_cache.invalidate_similar(embedding, threshold=0.70)
        assert invalidated == 1

        # Check key is now deleted
        assert await mock_redis.get(exact_key) is None

    async def test_invalidate_similar_miss(self, semantic_cache, mock_redis):
        """Test selective invalidation when similarity is below threshold."""
        request = AskRequest(query="test query")
        response = AskResponse(
            query="test query",
            answer="semantic answer",
            sources=[],
            chunks_used=1,
            search_mode="bm25",
        )
        embedding = [0.1] * 1024

        # Store
        await semantic_cache.store(request, response, embedding)

        # Confirm stored
        exact_key = semantic_cache._exact_key(request)
        assert await mock_redis.get(exact_key) is not None

        # Invalidate with dissimilar embedding
        dissimilar_embedding = [-0.1] * 1024
        invalidated = await semantic_cache.invalidate_similar(dissimilar_embedding, threshold=0.70)
        assert invalidated == 0

        # Check key is STILL present
        assert await mock_redis.get(exact_key) is not None

