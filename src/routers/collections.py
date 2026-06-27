import logging
from typing import List
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.dependencies import get_db_session
from src.models.user import User
from src.repositories.collection import CollectionRepository
from src.repositories.paper import PaperRepository
from src.schemas.api.collections import CollectionCreate, CollectionPaperAdd, CollectionResponse
from src.services.auth.rbac import require_researcher

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/collections", tags=["collections"])


@router.get("/", response_model=List[CollectionResponse])
async def list_collections(
    db: Session = Depends(get_db_session),
    current_user: User = Depends(require_researcher)
) -> List[CollectionResponse]:
    """Retrieve all collections belonging to current authenticated user."""
    user_id = getattr(current_user, "id", None)
    if not user_id:
        return []
    repo = CollectionRepository(db)
    return repo.get_all_by_user(user_id)


@router.post("/", response_model=CollectionResponse, status_code=status.HTTP_201_CREATED)
async def create_collection(
    request: CollectionCreate,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(require_researcher)
) -> CollectionResponse:
    """Create a new research paper collection."""
    user_id = getattr(current_user, "id", None)
    if not user_id:
        raise HTTPException(status_code=400, detail="Authentication required")
    repo = CollectionRepository(db)
    return repo.create(name=request.name, user_id=user_id, description=request.description)


@router.delete("/{collection_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_collection(
    collection_id: UUID,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(require_researcher)
):
    """Delete a collection."""
    repo = CollectionRepository(db)
    collection = repo.get_by_id(collection_id)
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")
    
    # Author check
    user_id = getattr(current_user, "id", None)
    if user_id and collection.user_id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this collection")

    repo.delete(collection_id)


@router.post("/{collection_id}/papers", status_code=status.HTTP_200_OK)
async def add_paper_to_collection(
    collection_id: UUID,
    request: CollectionPaperAdd,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(require_researcher)
):
    """Add a research paper (by DB UUID) to a collection."""
    repo = CollectionRepository(db)
    collection = repo.get_by_id(collection_id)
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")
        
    user_id = getattr(current_user, "id", None)
    if user_id and collection.user_id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to edit this collection")

    # Verify paper exists in DB
    paper_repo = PaperRepository(db)
    paper = paper_repo.get_by_id(request.paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found in local database")

    success = repo.add_paper_to_collection(collection_id, request.paper_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to add paper to collection")
    return {"message": "Paper added to collection successfully"}


@router.delete("/{collection_id}/papers/{paper_id}", status_code=status.HTTP_200_OK)
async def remove_paper_from_collection(
    collection_id: UUID,
    paper_id: UUID,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(require_researcher)
):
    """Remove a paper from a collection."""
    repo = CollectionRepository(db)
    collection = repo.get_by_id(collection_id)
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")
        
    user_id = getattr(current_user, "id", None)
    if user_id and collection.user_id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to edit this collection")

    success = repo.remove_paper_from_collection(collection_id, paper_id)
    if not success:
        raise HTTPException(status_code=404, detail="Association not found in collection")
    return {"message": "Paper removed from collection successfully"}


@router.get("/{collection_id}/papers")
async def get_collection_papers(
    collection_id: UUID,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(require_researcher)
):
    """Get all papers associated with a collection."""
    repo = CollectionRepository(db)
    collection = repo.get_by_id(collection_id)
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")
        
    user_id = getattr(current_user, "id", None)
    if user_id and collection.user_id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to view this collection")

    papers = repo.get_papers_in_collection(collection_id)
    
    # Return formatted list
    return [
        {
            "id": p.id,
            "arxiv_id": p.arxiv_id,
            "title": p.title,
            "authors": p.authors,
            "abstract": p.abstract,
            "categories": p.categories,
            "published_date": p.published_date,
            "pdf_url": p.pdf_url
        }
        for p in papers
    ]
