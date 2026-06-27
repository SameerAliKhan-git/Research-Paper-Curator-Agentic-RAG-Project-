from typing import List, Literal

from pydantic import BaseModel, Field


class SummarizeRequest(BaseModel):
    """Request model for paper summarization."""

    model: str = Field("llama3.2:1b", description="Ollama model to use for generation")
    summary_type: Literal["brief", "detailed", "technical"] = Field("brief", description="Type of summary to generate")

    class Config:
        json_schema_extra = {
            "example": {
                "model": "llama3.2:1b",
                "summary_type": "brief",
            }
        }


class SummarizeResponse(BaseModel):
    """Response model for paper summarization."""

    arxiv_id: str = Field(..., description="ArXiv identifier of the paper")
    title: str = Field(..., description="Title of the paper")
    summary: str = Field(..., description="Generated summary of the paper")
    key_findings: List[str] = Field(..., description="Key findings extracted from the paper")
    categories: List[str] = Field(..., description="arXiv categories of the paper")

    class Config:
        json_schema_extra = {
            "example": {
                "arxiv_id": "1706.03762",
                "title": "Attention Is All You Need",
                "summary": "This paper introduces the Transformer architecture...",
                "key_findings": [
                    "Self-attention mechanisms can replace recurrence entirely",
                    "The Transformer achieves state-of-the-art on WMT 2014 English-German translation",
                ],
                "categories": ["cs.CL", "cs.LG"],
            }
        }
