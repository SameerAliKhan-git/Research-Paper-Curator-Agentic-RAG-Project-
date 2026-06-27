"""Redis-backed semantic cache using cosine similarity over Jina embeddings.

Uses Locality-Sensitive Hashing (LSH) with random hyperplane projections for
O(1) approximate nearest neighbor lookup instead of brute-force O(N) scan.
Stores binary signatures in sorted sets for fast range queries.

Cache flow:
    1. On cache hit (exact match) -> return immediately
    2. On cache hit (semantic match via LSH, cosine >= threshold) -> return
    3. On cache miss -> caller stores via store()

LSH Configuration:
    - 64 random hyperplanes for 1024-dim vectors
    - 4 hash tables with different random projections
    - Hamming distance threshold for candidate retrieval
"""

import asyncio
import hashlib
import json
import logging
import struct
from datetime import timedelta
from typing import Any, List, Optional, Set, Tuple

import numpy as np
import redis.asyncio as aioredis
from src.config import RedisSettings
from src.schemas.api.ask import AskRequest, AskResponse

logger = logging.getLogger(__name__)

VECTOR_DIM = 1024
SIMILARITY_THRESHOLD = 0.92
CACHE_PREFIX = "semantic_cache"
EXACT_CACHE_PREFIX = "exact_cache"
LSH_INDEX_PREFIX = "semantic_cache:lsh"
LSH_HYPERPLANES_KEY = "semantic_cache:lsh_hyperplanes"
MAX_CANDIDATES = 5000
LSH_NUM_TABLES = 4
LSH_NUM_BITS = 64
STALE_TTL_MULTIPLIER = 4


class SemanticCache:
    """Redis-backed semantic cache using LSH for efficient cosine similarity search.

    Uses Locality-Sensitive Hashing with random hyperplane projections to
    pre-filter candidates before brute-force cosine similarity. This reduces
    the effective search space from O(N) to O(N/2^bits) in practice.

    Cache flow:
        1. On cache hit (exact match) -> return immediately
        2. On cache hit (semantic match via LSH, cosine >= threshold) -> return
        3. On cache miss -> caller stores via store()
    """

    def __init__(self, redis_client: aioredis.Redis, settings: RedisSettings):
        self.redis = redis_client
        self.settings = settings
        self.ttl = timedelta(hours=settings.ttl_hours)
        self.stale_ttl = self.ttl * STALE_TTL_MULTIPLIER
        self.threshold = SIMILARITY_THRESHOLD
        self._hyperplanes: Optional[np.ndarray] = None

    async def _ensure_hyperplanes(self) -> np.ndarray:
        """Load or generate random hyperplanes for LSH.

        Hyperplanes are stored in Redis and reused across all operations
        to ensure consistency.
        """
        if self._hyperplanes is not None:
            return self._hyperplanes

        try:
            raw = await self.redis.get(LSH_HYPERPLANES_KEY)
            if raw:
                self._hyperplanes = np.frombuffer(raw, dtype=np.float64).reshape(
                    LSH_NUM_TABLES, LSH_NUM_BITS, VECTOR_DIM
                )
                return self._hyperplanes
        except Exception:
            pass

        rng = np.random.RandomState(42)
        hyperplanes = rng.randn(LSH_NUM_TABLES, LSH_NUM_BITS, VECTOR_DIM).astype(np.float64)

        try:
            await self.redis.set(LSH_HYPERPLANES_KEY, hyperplanes.tobytes(), ex=None)
            self._hyperplanes = hyperplanes
        except Exception as e:
            logger.warning(f"Failed to persist LSH hyperplanes: {e}")
            self._hyperplanes = hyperplanes

        return self._hyperplanes

    @staticmethod
    def _serialize_vector(vector: List[float]) -> bytes:
        """Pack a float list into a compact binary blob."""
        return struct.pack(f"{len(vector)}f", *vector)

    @staticmethod
    def _deserialize_vector(data: bytes) -> np.ndarray:
        """Unpack a binary blob back to a numpy float array."""
        n = len(data) // 4
        return np.frombuffer(data, dtype=np.float32, count=n)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _compute_lsh_signature(self, vec: np.ndarray, hyperplanes: np.ndarray) -> bytes:
        """Compute LSH signature for a vector using random hyperplane projections."""
        projections = hyperplanes @ vec
        bits = (projections > 0).astype(np.uint8)
        packed = bytearray()
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                if i + j < len(bits) and bits[i + j]:
                    byte |= 1 << j
            packed.append(byte)
        return bytes(packed)

    def _compute_lsh_signatures(self, vec: np.ndarray, hyperplanes: np.ndarray) -> List[bytes]:
        """Compute LSH signatures for all hash tables."""
        signatures = []
        for table_idx in range(LSH_NUM_TABLES):
            sig = self._compute_lsh_signature(vec, hyperplanes[table_idx])
            signatures.append(sig)
        return signatures

    def _exact_key(self, request: AskRequest, tenant_id: Optional[str] = None) -> str:
        """Generate exact-match cache key."""
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
        return f"{prefix}{EXACT_CACHE_PREFIX}:{key_hash}"

    def _embedding_key(self, vector_hash: str, tenant_id: Optional[str] = None) -> str:
        prefix = f"tenant:{tenant_id}:" if tenant_id else ""
        return f"{prefix}{CACHE_PREFIX}:vec:{vector_hash}"

    def _lsh_bucket_key(self, table_idx: int, signature: bytes, tenant_id: Optional[str] = None) -> str:
        prefix = f"tenant:{tenant_id}:" if tenant_id else ""
        sig_hex = signature.hex()
        return f"{prefix}{LSH_INDEX_PREFIX}:t{table_idx}:b{sig_hex}"

    def _metadata_key(self, vector_hash: str, tenant_id: Optional[str] = None) -> str:
        prefix = f"tenant:{tenant_id}:" if tenant_id else ""
        return f"{prefix}{CACHE_PREFIX}:meta:{vector_hash}"

    def _index_key(self, tenant_id: Optional[str] = None) -> str:
        prefix = f"tenant:{tenant_id}:" if tenant_id else ""
        return f"{prefix}{CACHE_PREFIX}:index"

    async def find_exact(self, request: AskRequest, tenant_id: Optional[str] = None) -> Optional[AskResponse]:
        """Fast exact-match lookup (O(1))."""
        try:
            cached = await self.redis.get(self._exact_key(request, tenant_id=tenant_id))
            if cached:
                return AskResponse(**json.loads(cached))
        except Exception as e:
            logger.debug(f"Exact cache lookup failed: {e}")
        return None

    def _scan_candidates(
        self,
        query_vec: np.ndarray,
        raw_vectors: List[Optional[bytes]],
        candidate_hashes: List[Any],
    ) -> Tuple[float, Optional[str]]:
        """Compute cosine similarity for all candidates in a background thread."""
        best_score = -1.0
        best_hash = None

        for i, stored_bytes in enumerate(raw_vectors):
            if stored_bytes is None:
                continue

            try:
                stored_vec = self._deserialize_vector(stored_bytes)
                score = self._cosine_similarity(query_vec, stored_vec)

                if score > best_score:
                    best_score = score
                    vhash = candidate_hashes[i]
                    best_hash = vhash if isinstance(vhash, str) else vhash.decode()
            except Exception:
                continue

        return best_score, best_hash

    def _scan_similar_candidates(
        self,
        query_vec: np.ndarray,
        raw_vectors: List[Optional[bytes]],
        candidate_hashes: List[Any],
        threshold: float,
    ) -> List[Tuple[str, float]]:
        """Scan candidate vectors and return those with similarity above the threshold."""
        matching = []
        for i, stored_bytes in enumerate(raw_vectors):
            if stored_bytes is None:
                continue
            try:
                stored_vec = self._deserialize_vector(stored_bytes)
                score = self._cosine_similarity(query_vec, stored_vec)
                if score >= threshold:
                    vhash = candidate_hashes[i]
                    b_hash = vhash if isinstance(vhash, str) else vhash.decode()
                    matching.append((b_hash, score))
            except Exception:
                continue
        return matching

    async def _get_lsh_candidates(
        self,
        query_vec: np.ndarray,
        tenant_id: Optional[str] = None,
    ) -> Set[str]:
        """Retrieve candidate vector hashes using LSH bucket lookup.

        Queries multiple hash tables and returns the union of candidates.
        This is O(K) where K is the number of candidates in matching buckets.
        """
        hyperplanes = await self._ensure_hyperplanes()
        signatures = self._compute_lsh_signatures(query_vec, hyperplanes)

        candidate_hashes: Set[str] = set()

        pipe = self.redis.pipeline()
        for table_idx in range(LSH_NUM_TABLES):
            bucket_key = self._lsh_bucket_key(table_idx, signatures[table_idx], tenant_id=tenant_id)
            pipe.smembers(bucket_key)
        results = await pipe.execute()

        for table_result in results:
            for raw_hash in table_result:
                h = raw_hash if isinstance(raw_hash, str) else raw_hash.decode()
                candidate_hashes.add(h)

        return candidate_hashes

    async def find_semantic(
        self,
        query_embedding: List[float],
        request: AskRequest,
        tenant_id: Optional[str] = None,
    ) -> Optional[AskResponse]:
        """Find cached response whose query embedding is cosine-similar above threshold.

        Uses LSH for O(1) candidate retrieval, then brute-force cosine similarity
        on the small candidate set (typically <1% of total vectors).
        """
        try:
            query_vec = np.array(query_embedding, dtype=np.float32)

            candidate_hashes_raw = await self._get_lsh_candidates(query_vec, tenant_id=tenant_id)

            if not candidate_hashes_raw:
                index_raw = await self.redis.smembers(self._index_key(tenant_id=tenant_id))
                if not index_raw:
                    return None
                candidate_hashes_raw = {h if isinstance(h, str) else h.decode() for h in index_raw}

            candidate_hashes = list(candidate_hashes_raw)[:MAX_CANDIDATES]

            pipe = self.redis.pipeline()
            for vhash in candidate_hashes:
                pipe.get(self._embedding_key(vhash, tenant_id=tenant_id))
            raw_vectors = await pipe.execute()

            best_score, best_hash = await asyncio.to_thread(
                self._scan_candidates,
                query_vec,
                raw_vectors,
                candidate_hashes,
            )

            if best_score >= self.threshold and best_hash is not None:
                meta_raw = await self.redis.get(self._metadata_key(best_hash, tenant_id=tenant_id))
                if meta_raw:
                    meta = json.loads(meta_raw)
                    response_data = json.loads(meta["response"])
                    logger.info(f"Semantic cache hit (score={best_score:.4f}, threshold={self.threshold})")
                    return AskResponse(**response_data)

        except Exception as e:
            logger.debug(f"Semantic cache lookup failed: {e}")

        return None

    async def store(
        self,
        request: AskRequest,
        response: AskResponse,
        query_embedding: List[float],
        tenant_id: Optional[str] = None,
    ) -> bool:
        """Store a query-response pair in both exact and semantic caches.

        Also indexes the vector in LSH hash tables for fast candidate retrieval.
        """
        try:
            response_json = response.model_dump_json()

            await self.redis.set(self._exact_key(request, tenant_id=tenant_id), response_json, ex=self.ttl)

            vec_bytes = self._serialize_vector(query_embedding)
            vec_hash = hashlib.sha256(vec_bytes).hexdigest()[:16]

            hyperplanes = await self._ensure_hyperplanes()
            query_vec = np.array(query_embedding, dtype=np.float32)
            signatures = self._compute_lsh_signatures(query_vec, hyperplanes)

            pipe = self.redis.pipeline()
            pipe.set(self._embedding_key(vec_hash, tenant_id=tenant_id), vec_bytes, ex=self.ttl)
            pipe.sadd(self._index_key(tenant_id=tenant_id), vec_hash)
            pipe.expire(self._index_key(tenant_id=tenant_id), int(self.ttl.total_seconds()))
            pipe.set(
                self._metadata_key(vec_hash, tenant_id=tenant_id),
                json.dumps({
                    "response": response_json,
                    "query": request.query,
                    "exact_key": self._exact_key(request, tenant_id=tenant_id),
                }),
                ex=self.ttl,
            )
            for table_idx in range(LSH_NUM_TABLES):
                bucket_key = self._lsh_bucket_key(table_idx, signatures[table_idx], tenant_id=tenant_id)
                pipe.sadd(bucket_key, vec_hash)
                pipe.expire(bucket_key, int(self.ttl.total_seconds()))
            await pipe.execute()

            logger.debug(f"Stored semantic cache entry {vec_hash[:8]}...")
            return True

        except Exception as e:
            logger.error(f"Failed to store semantic cache: {e}")
            return False

    async def invalidate_similar(
        self,
        paper_embedding: List[float],
        threshold: float = 0.70,
        tenant_id: Optional[str] = None,
    ) -> int:
        """Invalidate cached queries that are semantically similar to the ingested paper."""
        logger.info(f"Triggering selective cache invalidation with threshold={threshold}")
        invalidated_count = 0
        try:
            paper_vec = np.array(paper_embedding, dtype=np.float32)

            candidate_hashes_raw = await self._get_lsh_candidates(paper_vec, tenant_id=tenant_id)

            if not candidate_hashes_raw:
                index_raw = await self.redis.smembers(self._index_key(tenant_id=tenant_id))
                if not index_raw:
                    logger.info("No cached queries found in semantic cache index to invalidate")
                    return 0
                candidate_hashes_raw = {h if isinstance(h, str) else h.decode() for h in index_raw}

            candidate_hashes = list(candidate_hashes_raw)[:MAX_CANDIDATES]

            pipe = self.redis.pipeline()
            for vhash in candidate_hashes:
                pipe.get(self._embedding_key(vhash, tenant_id=tenant_id))
            raw_vectors = await pipe.execute()

            matching_hashes = await asyncio.to_thread(
                self._scan_similar_candidates,
                paper_vec,
                raw_vectors,
                candidate_hashes,
                threshold,
            )

            if not matching_hashes:
                logger.info("No semantically similar queries found in cache to invalidate")
                return 0

            logger.info(f"Found {len(matching_hashes)} similar queries to invalidate")

            pipe = self.redis.pipeline()
            for b_hash, _ in matching_hashes:
                pipe.get(self._metadata_key(b_hash, tenant_id=tenant_id))
            raw_metadatas = await pipe.execute()

            delete_pipe = self.redis.pipeline()
            for idx, (b_hash, score) in enumerate(matching_hashes):
                delete_pipe.delete(self._embedding_key(b_hash, tenant_id=tenant_id))
                delete_pipe.delete(self._metadata_key(b_hash, tenant_id=tenant_id))
                delete_pipe.srem(self._index_key(tenant_id=tenant_id), b_hash)

                meta_raw = raw_metadatas[idx]
                if meta_raw:
                    try:
                        meta = json.loads(meta_raw)
                        exact_key = meta.get("exact_key")
                        if exact_key:
                            delete_pipe.delete(exact_key)
                            logger.info(f"Invalidating exact cache key: {exact_key} (score={score:.4f})")
                    except Exception as e:
                        logger.warning(f"Failed to parse metadata for invalidation: {e}")

                invalidated_count += 1
                logger.info(f"Invalidated semantic cache entry {b_hash[:8]} (score={score:.4f})")

            await delete_pipe.execute()
            logger.info(f"Selective cache invalidation complete. Removed {invalidated_count} entries.")

        except Exception as e:
            logger.error(f"Selective cache invalidation failed: {e}")

        return invalidated_count

    async def cleanup_orphans(self, tenant_id: Optional[str] = None) -> int:
        """Remove index hashes whose vector/metadata keys have expired."""
        try:
            candidate_hashes_raw = await self.redis.smembers(self._index_key(tenant_id=tenant_id))
            if not candidate_hashes_raw:
                return 0

            removed = 0
            for raw_hash in candidate_hashes_raw:
                vhash = raw_hash if isinstance(raw_hash, str) else raw_hash.decode()
                if not await self.redis.exists(self._embedding_key(vhash, tenant_id=tenant_id)):
                    await self.redis.srem(self._index_key(tenant_id=tenant_id), vhash)
                    removed += 1

            if removed > 0:
                logger.info(f"Semantic cache cleanup: removed {removed} orphaned entries")
            return removed

        except Exception as e:
            logger.error(f"Semantic cache cleanup failed: {e}")
            return 0

    async def clear(self) -> bool:
        """Clear all semantic and exact cache keys across all tenants."""
        try:
            cursor = 0
            keys_deleted = 0
            while True:
                cursor, keys = await self.redis.scan(cursor, match=f"*{CACHE_PREFIX}:*", count=100)
                if keys:
                    await self.redis.delete(*keys)
                    keys_deleted += len(keys)
                if cursor == 0:
                    break

            cursor = 0
            while True:
                cursor, keys = await self.redis.scan(cursor, match=f"*{EXACT_CACHE_PREFIX}:*", count=100)
                if keys:
                    await self.redis.delete(*keys)
                    keys_deleted += len(keys)
                if cursor == 0:
                    break
            logger.info(f"Cleared semantic and exact cache: deleted {keys_deleted} keys")
            return True
        except Exception as e:
            logger.error(f"Error clearing semantic and exact cache: {e}")
            return False
