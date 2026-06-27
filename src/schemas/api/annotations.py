from datetime import datetime
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, Field


class AnnotationCreate(BaseModel):
    paper_id: UUID = Field(description="UUID of the paper to annotate")
    note: str = Field(min_length=1, description="Content of the note/annotation")
    text_selection: Optional[str] = Field(default=None, description="Optional original selected text from PDF")
    tag: Optional[str] = Field(default=None, description="Optional tag (e.g. key result, methodology, limitation)")
    page: Optional[int] = Field(default=None, description="Optional page number inside the PDF")


class AnnotationUpdate(BaseModel):
    note: Optional[str] = Field(default=None, description="Updated content of the note/annotation")
    text_selection: Optional[str] = Field(default=None, description="Updated selected text")
    tag: Optional[str] = Field(default=None, description="Updated tag")
    page: Optional[int] = Field(default=None, description="Updated page number")


class AnnotationResponse(BaseModel):
    id: UUID
    paper_id: UUID
    user_id: UUID
    note: str
    text_selection: Optional[str]
    tag: Optional[str]
    page: Optional[int]
    created_at: datetime

    class Config:
        from_attributes = True
