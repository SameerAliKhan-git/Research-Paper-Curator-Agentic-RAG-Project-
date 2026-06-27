import asyncio
import logging
import time
from typing import List, Optional

import httpx
from src.schemas.embeddings.jina import JinaEmbeddingRequest, JinaEmbeddingResponse
from src.services.circuit_breaker import async_circuit_breaker_retry

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY_BASE = 1.0
JINA_MAX_BATCH_SIZE = 2048  # Jina API max per request


class JinaEmbeddingsClient:
    """Client for Jina AI embeddings API.

    Uses Jina embeddings v3 model with 1024 dimensions optimized for retrieval.
    Documentation: https://jina.ai/embeddings
    """

    def __init__(self, api_key: str, base_url: str = "https://api.jina.ai/v1", max_concurrency: int = 5):
        """Initialize Jina embeddings client.

        :param api_key: Jina API key
        :param base_url: API base URL
        :param max_concurrency: Max concurrent API requests
        """
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.client = httpx.AsyncClient(timeout=60.0)
        self._semaphore = asyncio.Semaphore(max_concurrency)
        logger.info(f"Jina embeddings client initialized (max_concurrency={max_concurrency})")

    @async_circuit_breaker_retry(
        service_name="jina_embeddings",
        max_retries=3,
        retry_exceptions=(httpx.HTTPStatusError, httpx.HTTPError),
        failure_threshold=5,
        recovery_timeout=30,
    )
    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a single batch of texts via Jina API."""
        request_data = JinaEmbeddingRequest(model="jina-embeddings-v3", task="retrieval.passage", dimensions=1024, input=texts)
        response = await self.client.post(f"{self.base_url}/embeddings", headers=self.headers, json=request_data.model_dump())
        response.raise_for_status()
        result = JinaEmbeddingResponse(**response.json())
        return [item["embedding"] for item in result.data]

    async def embed_passages(self, texts: List[str], batch_size: int = 2048) -> List[List[float]]:
        """Embed text passages for indexing with concurrent batching.

        :param texts: List of text passages to embed
        :param batch_size: Number of texts per API call (max 2048)
        :returns: List of embedding vectors in input order
        """
        if not self.api_key or self.api_key == "your_jina_api_key_here" or "your_jina_api_key" in self.api_key:
            logger.warning("Dummy or missing Jina API key. Skipping passages embedding.")
            return [[1.0] + [0.0] * 1023 for _ in range(len(texts))]

        batch_size = min(batch_size, JINA_MAX_BATCH_SIZE)
        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

        logger.info(f"Embedding {len(texts)} passages in {len(batches)} batch(es) (size={batch_size})")
        t0 = time.time()

        # Pre-allocate result list for correct ordering
        all_embeddings: List[List[float]] = [[] for _ in texts]
        results_map: dict[int, List[List[float]]] = {}

        async def _embed_batch_with_semaphore(batch_idx: int, batch: List[str]) -> List[List[float]]:
            async with self._semaphore:
                return await self._embed_batch(batch)

        tasks = [_embed_batch_with_semaphore(i, batch) for i, batch in enumerate(batches)]
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        for batch_idx, result in enumerate(completed):
            if isinstance(result, Exception):
                logger.error(f"Batch {batch_idx} failed: {result}")
                raise
            start = batch_idx * batch_size
            for j, emb in enumerate(result):
                all_embeddings[start + j] = emb

        elapsed = time.time() - t0
        logger.info(f"Embedded {len(texts)} passages in {elapsed:.1f}s")
        return all_embeddings

    @async_circuit_breaker_retry(
        service_name="jina_embeddings",
        max_retries=3,
        retry_exceptions=(httpx.HTTPStatusError, httpx.HTTPError),
        failure_threshold=5,
        recovery_timeout=30,
    )
    async def _embed_query_api(self, query: str) -> List[float]:
        """Call Jina API to embed query with retries/circuit breaker."""
        request_data = JinaEmbeddingRequest(model="jina-embeddings-v3", task="retrieval.query", dimensions=1024, input=[query])
        response = await self.client.post(f"{self.base_url}/embeddings", headers=self.headers, json=request_data.model_dump())
        response.raise_for_status()
        result = JinaEmbeddingResponse(**response.json())
        return result.data[0]["embedding"]

    async def embed_query(self, query: str) -> Optional[List[float]]:
        """Embed a search query.

        Falls back gracefully to BM25 (returning None) if the API key is invalid
        or the API call fails.

        :param query: Query text to embed
        :returns: Embedding vector for the query, or None if embeddings are unavailable
        """
        # If API key is dummy/placeholder, don't even make the request
        if not self.api_key or self.api_key == "your_jina_api_key_here" or "your_jina_api_key" in self.api_key:
            logger.warning("Dummy or missing Jina API key. Skipping vector embedding (will fallback to BM25 search).")
            return None

        try:
            embedding = await self._embed_query_api(query)
            logger.debug(f"Embedded query: '{query[:50]}...'")
            return embedding
        except Exception as e:
            logger.warning(
                f"Jina AI query embedding failed ({type(e).__name__}): {e}. "
                "Skipping vector embedding (will fallback to BM25 search)."
            )
            return None

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
