import logging
from typing import List
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.dependencies import get_db_session
from src.models.user import User
from src.repositories.annotation import AnnotationRepository
from src.repositories.paper import PaperRepository
from src.schemas.api.annotations import AnnotationCreate, AnnotationResponse, AnnotationUpdate
from src.services.auth.rbac import require_researcher

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/annotations", tags=["annotations"])


@router.post("/", response_model=AnnotationResponse, status_code=status.HTTP_201_CREATED)
async def create_annotation(
    request: AnnotationCreate,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(require_researcher)
) -> AnnotationResponse:
    """Create a new note/annotation on a specific paper."""
    user_id = getattr(current_user, "id", None)
    if not user_id:
        raise HTTPException(status_code=400, detail="Authentication required")

    # Verify paper exists
    paper_repo = PaperRepository(db)
    paper = paper_repo.get_by_id(request.paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    repo = AnnotationRepository(db)
    return repo.create(
        paper_id=request.paper_id,
        user_id=user_id,
        note=request.note,
        text_selection=request.text_selection,
        tag=request.tag,
        page=request.page
    )


@router.get("/paper/{paper_id}", response_model=List[AnnotationResponse])
async def list_paper_annotations(
    paper_id: UUID,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(require_researcher)
) -> List[AnnotationResponse]:
    """Retrieve all annotations/notes for a specific paper."""
    repo = AnnotationRepository(db)
    return repo.get_all_by_paper(paper_id)


@router.put("/{annotation_id}", response_model=AnnotationResponse)
async def update_annotation(
    annotation_id: UUID,
    request: AnnotationUpdate,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(require_researcher)
) -> AnnotationResponse:
    """Update an existing annotation/note."""
    repo = AnnotationRepository(db)
    annotation = repo.get_by_id(annotation_id)
    if not annotation:
        raise HTTPException(status_code=404, detail="Annotation not found")

    user_id = getattr(current_user, "id", None)
    if user_id and annotation.user_id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to edit this annotation")

    if request.note is not None:
        annotation.note = request.note
    if request.text_selection is not None:
        annotation.text_selection = request.text_selection
    if request.tag is not None:
        annotation.tag = request.tag
    if request.page is not None:
        annotation.page = request.page

    return repo.update(annotation)


@router.delete("/{annotation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_annotation(
    annotation_id: UUID,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(require_researcher)
):
    """Delete an annotation/note."""
    repo = AnnotationRepository(db)
    annotation = repo.get_by_id(annotation_id)
    if not annotation:
        raise HTTPException(status_code=404, detail="Annotation not found")

    user_id = getattr(current_user, "id", None)
    if user_id and annotation.user_id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this annotation")

    repo.delete(annotation_id)
