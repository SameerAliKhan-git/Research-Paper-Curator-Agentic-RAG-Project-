"""Unified OpenSearch client supporting both simple BM25 and hybrid search."""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from opensearchpy import OpenSearch, OpenSearchException
from src.config import Settings
from src.services.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError, get_circuit_breaker

from .index_config_hybrid import (
    ARXIV_PAPERS_CHUNKS_MAPPING,
    HYBRID_RRF_PIPELINE,
    ARXIV_PAPER_VISUAL_PAGES_INDEX,
    ARXIV_PAPER_VISUAL_PAGES_MAPPING,
)
from .query_builder import QueryBuilder

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY_BASE = 0.5


class OpenSearchClient:
    """OpenSearch client supporting BM25 and hybrid search with native RRF."""

    def __init__(self, host: str, settings: Settings):
        self.host = host
        self.settings = settings
        self.index_name = f"{settings.opensearch.index_name}-{settings.opensearch.chunk_index_suffix}"

        # Circuit breaker for OpenSearch operations
        self._circuit_breaker = get_circuit_breaker(
            "opensearch",
            failure_threshold=5,
            recovery_timeout=30.0,
        )

        # Configure SSL based on settings
        use_ssl = host.startswith("https")
        self.client = OpenSearch(
            hosts=[host],
            use_ssl=use_ssl,
            verify_certs=use_ssl,
            ssl_show_warn=False,
            timeout=int(settings.opensearch.request_timeout),
            max_retries=3,
            retry_on_timeout=True,
        )

        logger.info(f"OpenSearch client initialized with host: {host}")

    def health_check(self) -> bool:
        """Check if OpenSearch cluster is healthy."""
        try:
            health = self.client.cluster.health()
            return health["status"] in ["green", "yellow"]
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics for the hybrid index."""
        try:
            if not self.client.indices.exists(index=self.index_name):
                return {"index_name": self.index_name, "exists": False, "document_count": 0}

            stats_response = self.client.indices.stats(index=self.index_name)
            index_stats = stats_response["indices"][self.index_name]["total"]

            return {
                "index_name": self.index_name,
                "exists": True,
                "document_count": index_stats["docs"]["count"],
                "deleted_count": index_stats["docs"]["deleted"],
                "size_in_bytes": index_stats["store"]["size_in_bytes"],
            }

        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {"index_name": self.index_name, "exists": False, "document_count": 0, "error": str(e)}

    def setup_indices(self, force: bool = False) -> Dict[str, bool]:
        """Setup the hybrid search index and RRF pipeline."""
        results = {}
        results["hybrid_index"] = self._create_hybrid_index(force)
        results["rrf_pipeline"] = self._create_rrf_pipeline(force)
        results["visual_index"] = self._create_visual_index(force)
        return results

    def _create_visual_index(self, force: bool = False) -> bool:
        """Create visual page layout index for ColPali vision RAG.

        :param force: If True, recreate index even if it exists
        :returns: True if created, False if already exists
        """
        try:
            index_name = ARXIV_PAPER_VISUAL_PAGES_INDEX
            if force and self.client.indices.exists(index=index_name):
                self.client.indices.delete(index=index_name)
                logger.info(f"Deleted existing visual index: {index_name}")

            if not self.client.indices.exists(index=index_name):
                self.client.indices.create(index=index_name, body=ARXIV_PAPER_VISUAL_PAGES_MAPPING)
                logger.info(f"Created visual index: {index_name}")
                return True

            logger.info(f"Visual index already exists: {index_name}")
            return False

        except Exception as e:
            if "resource_already_exists_exception" in str(e):
                logger.info(f"Visual index already exists: {ARXIV_PAPER_VISUAL_PAGES_INDEX}")
                return False
            logger.error(f"Error creating visual index: {e}")
            raise

    def _create_hybrid_index(self, force: bool = False) -> bool:
        """Create hybrid index for all search types (BM25, vector, hybrid).

        :param force: If True, recreate index even if it exists
        :returns: True if created, False if already exists
        """
        try:
            if force and self.client.indices.exists(index=self.index_name):
                self.client.indices.delete(index=self.index_name)
                logger.info(f"Deleted existing hybrid index: {self.index_name}")

            if not self.client.indices.exists(index=self.index_name):
                self.client.indices.create(index=self.index_name, body=ARXIV_PAPERS_CHUNKS_MAPPING)
                logger.info(f"Created hybrid index: {self.index_name}")
                return True

            logger.info(f"Hybrid index already exists: {self.index_name}")
            return False

        except Exception as e:
            # Handle race condition when multiple workers start simultaneously:
            # all check exists() -> False, all try to create, only one succeeds.
            if "resource_already_exists_exception" in str(e):
                logger.info(f"Hybrid index already exists (created by another worker): {self.index_name}")
                return False
            logger.error(f"Error creating hybrid index: {e}")
            raise

    def _create_rrf_pipeline(self, force: bool = False) -> bool:
        """Create RRF search pipeline for native hybrid search.

        :param force: If True, recreate pipeline even if it exists
        :returns: True if created, False if already exists
        """
        try:
            pipeline_id = HYBRID_RRF_PIPELINE["id"]

            if force:
                try:
                    self.client.ingest.get_pipeline(id=pipeline_id)
                    self.client.ingest.delete_pipeline(id=pipeline_id)
                    logger.info(f"Deleted existing RRF pipeline: {pipeline_id}")
                except Exception:
                    pass

            try:
                self.client.ingest.get_pipeline(id=pipeline_id)
                logger.info(f"RRF pipeline already exists: {pipeline_id}")
                return False
            except Exception:
                pass
            pipeline_body = {
                "description": HYBRID_RRF_PIPELINE["description"],
                "phase_results_processors": HYBRID_RRF_PIPELINE["phase_results_processors"],
            }

            self.client.transport.perform_request("PUT", f"/_search/pipeline/{pipeline_id}", body=pipeline_body)

            logger.info(f"Created RRF search pipeline: {pipeline_id}")
            return True

        except Exception as e:
            logger.error(f"Error creating RRF pipeline: {e}")
            raise

    def search_papers(
        self,
        query: str,
        size: int = 10,
        from_: int = 0,
        categories: Optional[List[str]] = None,
        latest: bool = True,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """BM25 search for papers."""
        return self._search_bm25_only(
            query=query,
            size=size,
            from_=from_,
            categories=categories,
            latest=latest,
            tenant_id=tenant_id,
        )

    async def search_chunks_vector(
        self,
        query_embedding: List[float],
        size: int = 10,
        categories: Optional[List[str]] = None,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Pure vector search on chunks (async-safe).

        :param query_embedding: Query embedding vector
        :param size: Number of results
        :param categories: Optional category filter
        :param tenant_id: Optional tenant isolation identifier
        :returns: Search results
        """
        try:
            filter_clause = []
            if categories:
                filter_clause.append({"terms": {"categories": categories}})
            if tenant_id:
                filter_clause.append({"term": {"tenant_id": tenant_id}})

            knn_k = min(size, 10000)
            search_body = {
                "size": size,
                "query": {"knn": {"embedding": {"vector": query_embedding, "k": knn_k}}},
                "_source": {"excludes": ["embedding"]},
            }

            if filter_clause:
                search_body["query"] = {"bool": {"must": [search_body["query"]], "filter": filter_clause}}

            response = await asyncio.to_thread(self.client.search, index=self.index_name, body=search_body)

            results = {"total": response["hits"]["total"]["value"], "hits": []}

            for hit in response["hits"]["hits"]:
                chunk = hit["_source"]
                chunk["score"] = hit["_score"]
                chunk["chunk_id"] = hit["_id"]
                results["hits"].append(chunk)

            return results

        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return {"total": 0, "hits": []}

    async def search_unified(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        size: int = 10,
        from_: int = 0,
        categories: Optional[List[str]] = None,
        latest: bool = False,
        use_hybrid: bool = True,
        min_score: float = 0.0,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Unified search method supporting BM25, vector, and hybrid modes.

        :param query: Text query for search
        :param query_embedding: Optional embedding for vector/hybrid search
        :param size: Number of results to return
        :param from_: Offset for pagination
        :param categories: Optional category filter
        :param latest: Sort by date instead of relevance
        :param use_hybrid: If True and embedding provided, use hybrid search
        :param min_score: Minimum score threshold
        :param tenant_id: Optional tenant isolation identifier
        :returns: Search results
        """
        try:
            # If no embedding provided or hybrid disabled, use BM25 only
            if not query_embedding or not use_hybrid:
                return await self._search_bm25_only(
                    query=query, size=size, from_=from_, categories=categories, latest=latest, tenant_id=tenant_id
                )

            # Use native OpenSearch hybrid search with RRF pipeline
            return await self._search_hybrid_native(
                query=query,
                query_embedding=query_embedding,
                size=size,
                categories=categories,
                min_score=min_score,
                tenant_id=tenant_id,
            )

        except Exception as e:
            logger.error(f"Unified search error: {e}")
            return {"total": 0, "hits": []}

    async def _search_bm25_only(
        self,
        query: str,
        size: int,
        from_: int,
        categories: Optional[List[str]],
        latest: bool,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Pure BM25 search implementation with retry logic."""
        builder = QueryBuilder(
            query=query,
            size=size,
            from_=from_,
            categories=categories,
            latest_papers=latest,
            search_chunks=True,  # Enable chunk search mode
            tenant_id=tenant_id,
        )
        search_body = builder.build()

        for attempt in range(MAX_RETRIES):
            try:
                response = await asyncio.to_thread(self.client.search, index=self.index_name, body=search_body)
                break
            except OpenSearchException as e:
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY_BASE * (2**attempt)
                    logger.warning(f"OpenSearch search failed, retrying in {delay}s (attempt {attempt + 1})")
                    await asyncio.sleep(delay)
                    continue
                logger.error(f"OpenSearch search failed after {MAX_RETRIES} attempts: {e}")
                return {"total": 0, "hits": []}

        results = {"total": response["hits"]["total"]["value"], "hits": []}

        for hit in response["hits"]["hits"]:
            chunk = hit["_source"]
            chunk["score"] = hit["_score"]
            chunk["chunk_id"] = hit["_id"]

            if "highlight" in hit:
                chunk["highlights"] = hit["highlight"]

            results["hits"].append(chunk)

        logger.info(f"BM25 search for '{query[:50]}...' returned {results['total']} results")
        return results

    async def _search_hybrid_native(
        self,
        query: str,
        query_embedding: List[float],
        size: int,
        categories: Optional[List[str]],
        min_score: float,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Native OpenSearch hybrid search with RRF pipeline and retry logic."""
        builder = QueryBuilder(
            query=query,
            size=size * 2,
            from_=0,
            categories=categories,
            latest_papers=False,
            search_chunks=True,
            tenant_id=tenant_id,
        )
        bm25_search_body = builder.build()

        bm25_query = bm25_search_body["query"]

        knn_k = min(size * 2, 10000)
        hybrid_query = {"hybrid": {"queries": [bm25_query, {"knn": {"embedding": {"vector": query_embedding, "k": knn_k}}}]}}

        search_body = {
            "size": size,
            "query": hybrid_query,
            "_source": bm25_search_body["_source"],
            "highlight": bm25_search_body["highlight"],
        }

        # Execute search with RRF pipeline and retry logic
        for attempt in range(MAX_RETRIES):
            try:
                response = await asyncio.to_thread(
                    self.client.search,
                    index=self.index_name,
                    body=search_body,
                    params={"search_pipeline": HYBRID_RRF_PIPELINE["id"]},
                )
                break
            except OpenSearchException as e:
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY_BASE * (2**attempt)
                    logger.warning(f"OpenSearch hybrid search failed, retrying in {delay}s (attempt {attempt + 1})")
                    await asyncio.sleep(delay)
                    continue
                logger.error(f"OpenSearch hybrid search failed after {MAX_RETRIES} attempts: {e}")
                return {"total": 0, "hits": []}

        results = {"total": response["hits"]["total"]["value"], "hits": []}

        for hit in response["hits"]["hits"]:
            if hit["_score"] < min_score:
                continue

            chunk = hit["_source"]
            chunk["score"] = hit["_score"]
            chunk["chunk_id"] = hit["_id"]

            if "highlight" in hit:
                chunk["highlights"] = hit["highlight"]

            results["hits"].append(chunk)

        results["total"] = len(results["hits"])
        logger.info(f"Native hybrid search for '{query[:50]}...' returned {results['total']} results")
        return results

    async def search_chunks_hybrid(
        self,
        query: str,
        query_embedding: List[float],
        size: int = 10,
        categories: Optional[List[str]] = None,
        min_score: float = 0.0,
    ) -> Dict[str, Any]:
        """Hybrid search combining BM25 and vector similarity using native RRF."""
        return await self._search_hybrid_native(
            query=query, query_embedding=query_embedding, size=size, categories=categories, min_score=min_score
        )

    def index_chunk(self, chunk_data: Dict[str, Any], embedding: List[float]) -> bool:
        """Index a single chunk with its embedding.

        :param chunk_data: Chunk data dictionary
        :param embedding: Embedding vector
        :returns: True if successful
        """
        try:
            chunk_data["embedding"] = embedding

            response = self.client.index(index=self.index_name, body=chunk_data, refresh=False)

            return response["result"] in ["created", "updated"]

        except Exception as e:
            logger.error(f"Error indexing chunk: {e}")
            return False

    def bulk_index_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        """Bulk index multiple chunks with embeddings.

        :param chunks: List of dicts with 'chunk_data' and 'embedding'
        :returns: Statistics
        """
        from opensearchpy import helpers

        try:
            actions = []
            for chunk in chunks:
                chunk_data = chunk["chunk_data"].copy()
                chunk_data["embedding"] = chunk["embedding"]

                action = {"_index": self.index_name, "_source": chunk_data}
                actions.append(action)

            success, failed = helpers.bulk(self.client, actions, refresh=False)

            logger.info(f"Bulk indexed {success} chunks, {len(failed)} failed")
            return {"success": success, "failed": len(failed)}

        except Exception as e:
            logger.error(f"Bulk chunk indexing error: {e}")
            raise

    def delete_paper_chunks(self, arxiv_id: str) -> bool:
        """Delete all chunks for a specific paper.

        :param arxiv_id: ArXiv ID of the paper
        :returns: True if deletion was successful
        """
        try:
            response = self.client.delete_by_query(
                index=self.index_name, body={"query": {"term": {"arxiv_id": arxiv_id}}}, refresh=True
            )

            deleted = response.get("deleted", 0)
            logger.info(f"Deleted {deleted} chunks for paper {arxiv_id}")
            return deleted > 0

        except Exception as e:
            logger.error(f"Error deleting chunks: {e}")
            return False

    def get_chunks_by_paper(self, arxiv_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific paper.

        :param arxiv_id: ArXiv ID of the paper
        :returns: List of chunks sorted by chunk_index
        """
        try:
            search_body = {
                "query": {"term": {"arxiv_id": arxiv_id}},
                "size": 1000,
                "sort": [{"chunk_index": "asc"}],
                "_source": {"excludes": ["embedding"]},
            }

            response = self.client.search(index=self.index_name, body=search_body)

            chunks = []
            for hit in response["hits"]["hits"]:
                chunk = hit["_source"]
                chunk["chunk_id"] = hit["_id"]
                chunks.append(chunk)

            return chunks

        except Exception as e:
            logger.error(f"Error getting chunks: {e}")
            return []
