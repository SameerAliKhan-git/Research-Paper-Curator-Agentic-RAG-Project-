import time

from fastapi import APIRouter, Depends, HTTPException
from src.config import Settings, get_settings
from src.dependencies import APIKeyDep, OpenSearchDep, SessionDep
from src.repositories.paper import PaperRepository
from src.schemas.api.admin import (
    DashboardResponse,
    LLMStats,
    PapersStats,
    SearchStats,
    SystemStats,
)
from src.services.auth.api_key_service import APIKeyMetadata
from src.services.cost_tracker import cost_tracker

router = APIRouter(prefix="/admin", tags=["admin"])

_server_start_time = time.time()


def require_admin(key: APIKeyMetadata = Depends()) -> APIKeyMetadata:
    """Dependency that enforces admin-tier access."""
    if key.tier != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return key


@router.get("/dashboard", response_model=None)
async def get_dashboard(
    db: SessionDep,
    opensearch_client: OpenSearchDep,
    _key: APIKeyMetadata = Depends(require_admin),
    settings: Settings = Depends(get_settings),
) -> DashboardResponse:
    """Return aggregated system statistics for the admin dashboard."""
    repo = PaperRepository(db)
    total_papers = repo.get_count()
    processed_papers = repo.get_processed_count()
    processing_rate = (processed_papers / total_papers * 100) if total_papers > 0 else 0.0

    total_queries = 0
    cache_hit_rate = 0.0
    try:
        stats = opensearch_client.client.count(index=opensearch_client.index_name)
        total_queries = stats.get("count", 0)
    except Exception:
        pass

    token_stats = cost_tracker.get_total_tokens()
    total_tokens = token_stats["prompt_tokens"] + token_stats["completion_tokens"]

    return DashboardResponse(
        papers=PapersStats(
            total=total_papers,
            processed=processed_papers,
            processing_rate=round(processing_rate, 2),
        ),
        search=SearchStats(
            total_queries=total_queries,
            cache_hit_rate=round(cache_hit_rate, 2),
        ),
        llm=LLMStats(
            total_calls=len(cost_tracker._records),
            total_tokens=total_tokens,
            total_cost_usd=cost_tracker.get_total_cost(),
        ),
        system=SystemStats(
            uptime_seconds=round(time.time() - _server_start_time, 2),
            environment=settings.environment,
            version=settings.app_version,
        ),
    )
