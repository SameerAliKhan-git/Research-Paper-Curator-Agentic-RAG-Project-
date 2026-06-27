from typing import List

from pydantic import BaseModel, Field


class CitationNode(BaseModel):
    """A single node in the citation graph."""

    id: str = Field(..., description="ArXiv ID of the paper")
    title: str = Field(..., description="Title of the paper")


class CitationEdge(BaseModel):
    """A directed edge in the citation graph."""

    source: str = Field(..., description="ArXiv ID of the citing paper")
    target: str = Field(..., description="ArXiv ID of the cited paper")


class CitationGraph(BaseModel):
    """Citation graph structure for a paper."""

    paper_id: str = Field(..., description="ArXiv ID of the central paper")
    nodes: List[CitationNode] = Field(..., description="Papers in the graph")
    edges: List[CitationEdge] = Field(..., description="Citation relationships")

    class Config:
        json_schema_extra = {
            "example": {
                "paper_id": "1706.03762",
                "nodes": [
                    {"id": "1706.03762", "title": "Attention Is All You Need"},
                    {"id": "1409.0473", "title": "Neural Machine Translation by Jointly Learning to Align and Translate"},
                ],
                "edges": [
                    {"source": "1706.03762", "target": "1409.0473"},
                ],
            }
        }
