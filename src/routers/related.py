import logging

from fastapi import APIRouter, HTTPException
from src.dependencies import APIKeyDep, OpenSearchDep
from src.schemas.api.related import RelatedPaper, RelatedResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/papers", tags=["papers"])


@router.get("/{arxiv_id}/related", response_model=RelatedResponse)
async def get_related_papers(
    arxiv_id: str,
    opensearch_client: OpenSearchDep,
    _key: APIKeyDep = None,
) -> RelatedResponse:
    """Find papers related to a given paper using vector similarity search."""
    try:
        search_body = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"arxiv_id": arxiv_id}},
                    ]
                }
            },
            "size": 1,
            "_source": {"includes": ["embedding"]},
        }

        response = opensearch_client.client.search(
            index=opensearch_client.index_name,
            body=search_body,
        )

        hits = response.get("hits", {}).get("hits", [])
        if not hits:
            raise HTTPException(status_code=404, detail=f"Paper {arxiv_id} not found in index")

        paper_embedding = hits[0]["_source"].get("embedding")
        if not paper_embedding:
            raise HTTPException(status_code=404, detail=f"No embedding found for paper {arxiv_id}")

        vector_search_body = {
            "size": 6,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": paper_embedding,
                        "k": 6,
                    }
                }
            },
            "_source": {
                "includes": ["arxiv_id", "title", "abstract"],
                "excludes": ["embedding"],
            },
        }

        vector_response = opensearch_client.client.search(
            index=opensearch_client.index_name,
            body=vector_search_body,
        )

        related = []
        for hit in vector_response.get("hits", {}).get("hits", []):
            source = hit["_source"]
            if source.get("arxiv_id") == arxiv_id:
                continue

            related.append(
                RelatedPaper(
                    arxiv_id=source.get("arxiv_id", ""),
                    title=source.get("title", ""),
                    similarity_score=round(hit["_score"], 4),
                    abstract=source.get("abstract", ""),
                )
            )

        related = related[:5]

        return RelatedResponse(
            arxiv_id=arxiv_id,
            related_papers=related,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Related papers lookup failed for {arxiv_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to find related papers")
