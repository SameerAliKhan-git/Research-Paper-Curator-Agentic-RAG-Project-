from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field
from uuid import UUID

class InterestCreate(BaseModel):
    keyword: str = Field(..., description="Interest keyword or research area", min_length=2, max_length=100)

class InterestResponse(BaseModel):
    id: UUID
    keyword: str
    created_at: datetime

    class Config:
        from_attributes = True

class BriefingResponse(BaseModel):
    id: UUID
    arxiv_id: str
    title: str
    summary: str
    score: float
    published_date: datetime
    created_at: datetime

    class Config:
        from_attributes = True

class RecommendationTriggerRequest(BaseModel):
    target_date: Optional[str] = Field(None, description="Target date in YYYYMMDD format (defaults to yesterday)")
