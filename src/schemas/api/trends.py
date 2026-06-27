from typing import List

from pydantic import BaseModel, Field


class TrendDataPoint(BaseModel):
    """A single data point in the trends time series."""

    month: str = Field(..., description="Month in YYYY-MM format")
    category: str = Field(..., description="arXiv category")
    count: int = Field(..., description="Number of papers in that category for that month")


class TrendsResponse(BaseModel):
    """Response model for research trends."""

    data: List[TrendDataPoint] = Field(..., description="Monthly paper counts by category")
    total_papers: int = Field(..., description="Total papers across all data points")

    class Config:
        json_schema_extra = {
            "example": {
                "data": [
                    {"month": "2026-05", "category": "cs.AI", "count": 42},
                    {"month": "2026-05", "category": "cs.LG", "count": 31},
                    {"month": "2026-04", "category": "cs.AI", "count": 38},
                ],
                "total_papers": 111,
            }
        }


class PopularQuery(BaseModel):
    query: str = Field(..., description="Query search text")
    count: int = Field(..., description="Number of searches recorded")


class PopularQueriesResponse(BaseModel):
    queries: List[PopularQuery]


class IngestionStatsResponse(BaseModel):
    total_papers: int
    processed_papers: int
    papers_with_text: int
    processing_rate: float
    text_extraction_rate: float


class TopPaper(BaseModel):
    id: str
    arxiv_id: str
    title: str
    published_date: str
    categories: List[str]


class TopPapersResponse(BaseModel):
    papers: List[TopPaper]

