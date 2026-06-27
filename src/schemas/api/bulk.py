from typing import List, Optional

from pydantic import BaseModel, Field


class BulkImportRequest(BaseModel):
    """Request model for bulk paper import."""

    arxiv_ids: Optional[List[str]] = Field(None, min_length=1, max_length=50, description="List of arXiv paper IDs to ingest")
    category: Optional[str] = Field(None, description="arXiv category to fetch recent papers from (e.g. cs.AI)")

    class Config:
        json_schema_extra = {
            "example": {
                "arxiv_ids": ["2301.07041", "2302.08814"],
            }
        }


class BulkImportResponse(BaseModel):
    """Response model for bulk paper import."""

    task_id: str = Field(..., description="Tracking ID for the bulk import task")
    total_submitted: int = Field(..., description="Number of papers submitted for ingestion")
    status: str = Field("processing", description="Current status of the bulk import")

    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "abc123-def456",
                "total_submitted": 2,
                "status": "processing",
            }
        }
