import io
import logging
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from src.dependencies import get_db_session
from src.models.user import User
from src.repositories.collection import CollectionRepository
from src.services.auth.rbac import require_researcher
from src.services.sync.obsidian import export_collection_as_obsidian_zip
from src.services.sync.notion import NotionSyncService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sync", tags=["sync"])


class NotionSyncRequest(BaseModel):
    notion_token: str = Field(..., description="Notion Integration API token")
    database_id: str = Field(..., description="Notion Database UUID where pages will be inserted")


@router.get("/obsidian/{collection_id}")
async def sync_collection_obsidian(
    collection_id: UUID,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(require_researcher),
):
    """Export a paper collection as an Obsidian vault ZIP package."""
    col_repo = CollectionRepository(db)
    collection = col_repo.get_by_id(collection_id)
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")

    # Access control
    user_id = getattr(current_user, "id", None)
    if user_id and collection.user_id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to access this collection")

    papers = col_repo.get_papers_in_collection(collection_id)
    if not papers:
        raise HTTPException(status_code=400, detail="Cannot export an empty collection")

    zip_bytes = export_collection_as_obsidian_zip(collection.name, papers)
    
    filename = f"obsidian_vault_{collection.name.replace(' ', '_')}.zip"
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.post("/notion/{collection_id}", status_code=status.HTTP_200_OK)
async def sync_collection_notion(
    collection_id: UUID,
    request: NotionSyncRequest,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(require_researcher),
):
    """Synchronize papers in a collection directly into a Notion Database."""
    col_repo = CollectionRepository(db)
    collection = col_repo.get_by_id(collection_id)
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")

    user_id = getattr(current_user, "id", None)
    if user_id and collection.user_id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to access this collection")

    papers = col_repo.get_papers_in_collection(collection_id)
    if not papers:
        raise HTTPException(status_code=400, detail="Cannot sync an empty collection")

    notion_service = NotionSyncService(auth_token=request.notion_token)
    sync_result = await notion_service.sync_collection(
        database_id=request.database_id,
        papers=papers
    )
    return sync_result
