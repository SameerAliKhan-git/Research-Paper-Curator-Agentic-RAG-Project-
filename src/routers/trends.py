import logging
from typing import Optional, List
from fastapi import APIRouter, Query, Depends

from src.dependencies import SessionDep, get_cache_client, CacheDep
from src.repositories.paper import PaperRepository
from src.schemas.api.trends import (
    TrendDataPoint,
    TrendsResponse,
    PopularQueriesResponse,
    PopularQuery,
    IngestionStatsResponse,
    TopPapersResponse,
    TopPaper,
)
from src.services.query_tracker import QueryTracker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/trends", tags=["trends"])


@router.get("/", response_model=TrendsResponse)
async def get_trends(
    db: SessionDep,
    category: Optional[str] = Query(None, description="Filter by arXiv category"),
    months_back: int = Query(12, ge=1, le=60, description="Number of months to look back"),
) -> TrendsResponse:
    """Return monthly paper counts by category over time."""
    repo = PaperRepository(db)
    data = repo.get_category_trends(category=category, months_back=months_back)

    points = [TrendDataPoint(**row) for row in data]
    total = sum(p.count for p in points)

    return TrendsResponse(data=points, total_papers=total)


@router.get("/popular-queries", response_model=PopularQueriesResponse)
async def get_popular_queries(
    cache_client: CacheDep,
    limit: int = Query(10, ge=1, le=50)
) -> PopularQueriesResponse:
    """Retrieve the top search terms query volume recorded in Redis."""
    redis_client = cache_client.redis if cache_client else None
    tracker = QueryTracker(redis_client)
    popular = await tracker.get_popular_queries(limit=limit)
    
    queries = [PopularQuery(query=q, count=c) for q, c in popular]
    return PopularQueriesResponse(queries=queries)


@router.get("/top-papers", response_model=TopPapersResponse)
async def get_top_papers(db: SessionDep) -> TopPapersResponse:
    """Retrieve the top recently indexed papers."""
    repo = PaperRepository(db)
    papers = repo.get_processed_papers(limit=10)
    
    top = [
        TopPaper(
            id=str(p.id),
            arxiv_id=p.arxiv_id,
            title=p.title,
            published_date=p.published_date.strftime("%Y-%m-%d") if p.published_date else "unknown",
            categories=p.categories if isinstance(p.categories, list) else [str(p.categories)]
        )
        for p in papers
    ]
    return TopPapersResponse(papers=top)


@router.get("/ingestion-stats", response_model=IngestionStatsResponse)
async def get_ingestion_stats(db: SessionDep) -> IngestionStatsResponse:
    """Retrieve database indexing and parsing rate statistics."""
    repo = PaperRepository(db)
    stats = repo.get_processing_stats()
    return IngestionStatsResponse(**stats)
