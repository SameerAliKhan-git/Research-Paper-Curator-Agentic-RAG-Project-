import logging
from typing import List, Tuple, Optional
import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

POPULAR_QUERIES_KEY = "trends:popular_queries"


class QueryTracker:
    """Service to track query search volume trends in Redis."""

    def __init__(self, redis_client: Optional[aioredis.Redis]):
        self.redis = redis_client

    async def record_query(self, query: str) -> None:
        """Increment search frequency count for a given query in Redis."""
        if not self.redis or not query or not query.strip():
            return
        
        query_clean = query.strip().lower()
        # Avoid tracking excessively long queries or system prompts
        if len(query_clean) > 150:
            return

        try:
            # zincrby increments query count in sorted set
            await self.redis.zincrby(POPULAR_QUERIES_KEY, 1.0, query_clean)
        except Exception as e:
            logger.warning(f"Failed to record query in tracker: {e}")

    async def get_popular_queries(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Retrieve the top search terms sorted by frequency."""
        if not self.redis:
            return []

        try:
            # Get members sorted by score descending
            results = await self.redis.zrevrange(POPULAR_QUERIES_KEY, 0, limit - 1, withscores=True)
            return [(str(query), int(score)) for query, score in results]
        except Exception as e:
            logger.warning(f"Failed to retrieve popular queries: {e}")
            return []
