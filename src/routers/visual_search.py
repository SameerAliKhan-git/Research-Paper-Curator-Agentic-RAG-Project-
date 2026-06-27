import logging
from typing import List, Optional, Dict
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.dependencies import OpenSearchDep, EmbeddingsDep, TenantDep

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/visual-search", tags=["visual-search"])

class VisualSearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 4

class VisualSearchHit(BaseModel):
    arxiv_id: str
    page_number: int
    image_path: str
    score: float
    page_text: str
    layout_stats: Dict[str, int]

class VisualSearchResponse(BaseModel):
    query: str
    hits: List[VisualSearchHit]

@router.post("/query", response_model=VisualSearchResponse)
async def query_visual_pages(
    request: VisualSearchRequest,
    opensearch_client: OpenSearchDep,
    embeddings_service: EmbeddingsDep,
    tenant: TenantDep,
) -> VisualSearchResponse:
    """Query the visual page layout index using simulated ColPali embeddings."""
    try:
        from src.services.vision.colpali import ColPaliVisionService
        
        colpali_service = ColPaliVisionService(opensearch_client, embeddings_service)
        hits = await colpali_service.search_visual_pages(
            query=request.query,
            top_k=request.top_k or 4,
            tenant_id=tenant.tenant_id,
        )
        
        # Format response hits
        formatted_hits = []
        for hit in hits:
            formatted_hits.append(
                VisualSearchHit(
                    arxiv_id=hit["arxiv_id"],
                    page_number=hit["page_number"],
                    image_path=hit["image_path"],
                    score=hit["score"],
                    page_text=hit["page_text"],
                    layout_stats=hit["layout_stats"],
                )
            )
            
        return VisualSearchResponse(query=request.query, hits=formatted_hits)
        
    except Exception as e:
        logger.error(f"Failed to query visual page index: {e}")
        raise HTTPException(status_code=500, detail=f"Visual search failed: {e}")
