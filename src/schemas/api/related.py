from typing import List

from pydantic import BaseModel, Field


class RelatedPaper(BaseModel):
    """A single related paper."""

    arxiv_id: str = Field(..., description="ArXiv identifier of the related paper")
    title: str = Field(..., description="Title of the related paper")
    similarity_score: float = Field(..., description="Cosine similarity score")
    abstract: str = Field(..., description="Abstract of the related paper")


class RelatedResponse(BaseModel):
    """Response model for related papers endpoint."""

    arxiv_id: str = Field(..., description="ArXiv identifier of the source paper")
    related_papers: List[RelatedPaper] = Field(..., description="Top related papers sorted by similarity")
