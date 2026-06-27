import asyncio
import logging
import json
import hashlib
from typing import List, Optional
from urllib.parse import urlparse
from langchain_core.documents import Document
from duckduckgo_search import DDGS
import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

class WebSearchService:
    """Service for executing and caching web searches with relevance filtering and domain deduplication."""

    def __init__(self, redis_client: Optional[aioredis.Redis] = None, cache_ttl: int = 3600):
        self.redis = redis_client
        self.cache_ttl = cache_ttl

    def _get_domain(self, url: str) -> str:
        """Extract domain name from URL."""
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower()
        except Exception:
            return ""

    def _score_relevance(self, query: str, title: str, body: str) -> float:
        """Simple keyword overlap score between query terms and document fields."""
        query_words = set(query.lower().split())
        if not query_words:
            return 0.0
        
        content = (title + " " + body).lower()
        match_count = sum(1 for word in query_words if word in content)
        return match_count / len(query_words)

    async def search(self, query: str, max_results: int = 5) -> List[Document]:
        """Perform search with deduplication, scoring, truncation, and Redis caching."""
        if not query or not query.strip():
            return []

        query = query.strip()
        cache_key = f"web_search:{hashlib.sha256(query.encode('utf-8')).hexdigest()[:16]}"
        
        # 1. Try Redis cache
        if self.redis:
            try:
                cached_data = await self.redis.get(cache_key)
                if cached_data:
                    logger.info(f"Web search cache hit for: {query}")
                    results = json.loads(cached_data)
                    return [
                        Document(
                            page_content=res["page_content"],
                            metadata=res["metadata"]
                        )
                        for res in results
                    ]
            except Exception as e:
                logger.warning(f"Failed to read web search cache: {e}")

        # 2. Execute live search
        logger.info(f"Executing live web search for: {query}")
        raw_results = []
        try:
            def _ddg_search():
                with DDGS() as ddgs:
                    return list(ddgs.text(query, max_results=max_results * 3)) # Fetch extra for filtering
            
            raw_results = await asyncio.to_thread(_ddg_search)
        except Exception as e:
            logger.error(f"Live web search failed: {e}")
            return []

        # 3. Process, deduplicate, score, and truncate results
        domain_counts = {}
        processed_results = []

        for res in raw_results:
            title = res.get("title", "")
            body = res.get("body", "")
            url = res.get("href", "")
            if not url:
                continue

            # Deduplication: Max 2 results per domain
            domain = self._get_domain(url)
            if domain:
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
                if domain_counts[domain] > 2:
                    continue

            # Relevance scoring
            score = self._score_relevance(query, title, body)
            
            # Snippet extraction: truncate to 500 chars
            truncated_body = body[:500] + "..." if len(body) > 500 else body

            processed_results.append({
                "page_content": truncated_body,
                "metadata": {
                    "arxiv_id": "web",
                    "title": title,
                    "authors": ["Web Search"],
                    "score": score,
                    "source": url,
                    "section": "Web Snippet",
                    "search_mode": "web_search",
                }
            })

        # Sort by relevance score descending
        processed_results.sort(key=lambda x: x["metadata"]["score"], reverse=True)
        top_results = processed_results[:max_results]

        # 4. Save to Redis cache
        if self.redis and top_results:
            try:
                await self.redis.setex(
                    cache_key,
                    self.cache_ttl,
                    json.dumps(top_results)
                )
                logger.debug(f"Cached {len(top_results)} web search results for 1 hour")
            except Exception as e:
                logger.warning(f"Failed to cache web search results: {e}")

        # Convert back to Documents
        return [
            Document(
                page_content=res["page_content"],
                metadata=res["metadata"]
            )
            for res in top_results
        ]
