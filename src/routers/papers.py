from fastapi import APIRouter, Depends
from typing import List, Optional
from src.dependencies import SessionDep
from src.repositories.paper import PaperRepository

router = APIRouter(prefix="/papers", tags=["papers"])

@router.get("/")
def list_papers(limit: int = 10, offset: int = 0, db: SessionDep = None):
    """
    List all indexed papers from PostgreSQL database.
    """
    repo = PaperRepository(db)
    papers = repo.get_all(limit=limit, offset=offset)
    
    results = []
    for paper in papers:
        # Convert authors to list if stored as string, or handle json list
        authors_val = paper.authors
        if isinstance(authors_val, str):
            authors_val = [authors_val]
            
        results.append({
            "id": str(paper.id),
            "arxiv_id": paper.arxiv_id,
            "title": paper.title,
            "authors": authors_val,
            "abstract": paper.abstract,
            "categories": paper.categories if isinstance(paper.categories, list) else [paper.categories],
            "published_date": paper.published_date.isoformat() if paper.published_date else None,
            "pdf_url": paper.pdf_url,
            "pdf_processed": paper.pdf_processed
        })
    return results
