from typing import List, Optional

from pydantic import BaseModel, Field


class ReviewSection(BaseModel):
    """A section of the synthesized literature review."""

    heading: str = Field(..., description="Section heading")
    content: str = Field(..., description="Section content")


class ReviewPaper(BaseModel):
    """A paper referenced in the literature review."""

    arxiv_id: str = Field(..., description="ArXiv identifier")
    title: str = Field(..., description="Paper title")
    key_findings: List[str] = Field(default_factory=list, description="Extracted key findings")


class ReviewRequest(BaseModel):
    """Request model for literature review generation."""

    topic: str = Field(..., description="Research topic to review", min_length=1, max_length=500)
    category: Optional[str] = Field(None, description="Optional arXiv category filter")
    model: str = Field("llama3.2:1b", description="Ollama model to use for generation")

    class Config:
        json_schema_extra = {
            "example": {
                "topic": "transformer architectures for time series forecasting",
                "category": "cs.LG",
                "model": "llama3.2:1b",
            }
        }


class ReviewResponse(BaseModel):
    """Response model for literature review."""

    topic: str = Field(..., description="Original topic query")
    sections: List[ReviewSection] = Field(..., description="Synthesized review sections")
    papers: List[ReviewPaper] = Field(..., description="Papers used in the review")

    class Config:
        json_schema_extra = {
            "example": {
                "topic": "transformer architectures for time series forecasting",
                "sections": [
                    {
                        "heading": "Overview",
                        "content": "Recent work has applied transformer architectures to time series...",
                    }
                ],
                "papers": [
                    {
                        "arxiv_id": "2202.07125",
                        "title": "Informer: Beyond Efficient Transformer",
                        "key_findings": ["ProbSparse attention reduces complexity"],
                    }
                ],
            }
        }
