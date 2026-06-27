from pydantic import BaseModel, Field


class PapersStats(BaseModel):
    """Paper processing statistics."""

    total: int = Field(..., description="Total papers in database")
    processed: int = Field(..., description="Papers with PDF processed")
    processing_rate: float = Field(..., description="Percentage of processed papers")


class SearchStats(BaseModel):
    """Search statistics."""

    total_queries: int = Field(..., description="Total search queries executed")
    cache_hit_rate: float = Field(..., description="Cache hit rate percentage")


class LLMStats(BaseModel):
    """LLM usage statistics."""

    total_calls: int = Field(..., description="Total LLM API calls")
    total_tokens: int = Field(..., description="Total tokens consumed")
    total_cost_usd: float = Field(..., description="Total cost in USD")


class SystemStats(BaseModel):
    """System information."""

    uptime_seconds: float = Field(..., description="Server uptime in seconds")
    environment: str = Field(..., description="Deployment environment")
    version: str = Field(..., description="Application version")


class DashboardResponse(BaseModel):
    """Admin dashboard response model."""

    papers: PapersStats
    search: SearchStats
    llm: LLMStats
    system: SystemStats
