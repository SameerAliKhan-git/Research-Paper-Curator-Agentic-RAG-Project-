from datetime import datetime
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, Field


class CollectionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100, description="Name of the collection")
    description: Optional[str] = Field(default=None, description="Optional description of the collection")


class CollectionResponse(BaseModel):
    id: UUID
    name: str
    description: Optional[str]
    user_id: UUID
    created_at: datetime

    class Config:
        from_attributes = True


class CollectionPaperAdd(BaseModel):
    paper_id: UUID = Field(description="UUID of the paper to add/remove")
