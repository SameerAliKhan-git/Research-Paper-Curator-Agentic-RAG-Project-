import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from src.dependencies import (
    APIKeyDep,
    ArxivDep,
    EmbeddingsDep,
    LangfuseDep,
    OpenSearchDep,
    SessionDep,
)
from src.schemas.api.bulk import BulkImportRequest, BulkImportResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/papers", tags=["papers"])


@dataclass
class TaskStatus:
    task_id: str
    total: int
    completed: int = 0
    failed: int = 0
    status: str = "processing"
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    errors: list[str] = field(default_factory=list)


# In-memory task store (acceptable for single-process deployment)
_task_store: dict[str, TaskStatus] = {}


def _get_task_status(task_id: str) -> Optional[TaskStatus]:
    return _task_store.get(task_id)


async def _ingest_single_paper(
    arxiv_id: str,
    opensearch_client,
    embeddings_service,
    database,
):
    """Ingest a single paper using a shared database instance."""
    from src.repositories.paper import PaperRepository
    from src.services.arxiv.factory import make_arxiv_client
    from src.services.pdf_parser.factory import make_pdf_parser_service

    try:
        with database.get_session() as session:
            arxiv_client = make_arxiv_client()
            pdf_parser = make_pdf_parser_service()

            paper = await arxiv_client.fetch_paper_by_id(arxiv_id)
            if not paper:
                logger.error(f"Paper not found: {arxiv_id}")
                return

            title = paper.title
            abstract = paper.abstract
            content_hash = hashlib.sha256((title + abstract).encode("utf-8")).hexdigest()

            repo = PaperRepository(session)
            existing = repo.get_by_content_hash(content_hash)
            if existing:
                logger.info(f"Duplicate paper skipped (content_hash={content_hash[:12]}...): {arxiv_id}")
                return

            # Download PDF
            pdf_path = await arxiv_client.download_pdf(paper)
            pdf_content = None
            if pdf_path:
                try:
                    pdf_content = await pdf_parser.parse_pdf(pdf_path)
                except Exception as parse_err:
                    logger.error(f"Failed to parse PDF for {arxiv_id}: {parse_err}")

            # Index chunks in OpenSearch if PDF parsing succeeded
            if pdf_content:
                try:
                    from src.services.indexing.factory import make_hybrid_indexing_service

                    indexer = make_hybrid_indexing_service()
                    paper_id = existing.id if existing else uuid.uuid4()

                    # Parse published_date string to datetime
                    pub_date_str = paper.published_date
                    try:
                        from datetime import datetime

                        pub_date = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
                    except Exception:
                        pub_date = datetime.now()

                    paper_data = {
                        "id": str(paper_id),
                        "arxiv_id": arxiv_id,
                        "title": title,
                        "abstract": abstract,
                        "raw_text": pdf_content.raw_text,
                        "sections": [s.model_dump() for s in pdf_content.sections],
                        "authors": paper.authors,
                        "categories": paper.categories,
                        "published_date": pub_date.isoformat(),
                    }
                    await indexer.index_paper(paper_data)

                    # Process and index multimodal figures
                    try:
                        from src.services.multimodal import MultiModalProcessor
                        from src.services.ollama.factory import make_ollama_client
                        
                        ollama_client = make_ollama_client()
                        multimodal_processor = MultiModalProcessor()
                        
                        logger.info(f"Processing figures for bulk paper {arxiv_id}...")
                        await multimodal_processor.process_and_index_pdf_figures(
                            pdf_path=pdf_path,
                            paper_id=str(paper_id),
                            opensearch_client=indexer.opensearch_client,
                            ollama_client=ollama_client,
                            embeddings_client=indexer.embeddings_client,
                        )
                    except Exception as mm_err:
                        logger.warning(f"Failed to process multimodal figures for {arxiv_id}: {mm_err}")
                except Exception as index_err:
                    logger.error(f"Failed to index chunks in OpenSearch for {arxiv_id}: {index_err}")

            from datetime import datetime

            from src.schemas.arxiv.paper import PaperCreate

            # Parse published_date string to datetime
            pub_date = None
            if paper.published_date:
                try:
                    pub_date = datetime.fromisoformat(paper.published_date.replace("Z", "+00:00"))
                except Exception as ex:
                    logger.warning(f"Could not parse published_date '{paper.published_date}': {ex}")
                    pub_date = datetime.now()
            else:
                pub_date = datetime.now()

            paper_create = PaperCreate(
                arxiv_id=arxiv_id,
                title=title,
                authors=paper.authors,
                abstract=abstract,
                categories=paper.categories,
                published_date=pub_date,
                pdf_url=paper.pdf_url,
                raw_text=pdf_content.raw_text if pdf_content else None,
                sections=[s.model_dump() for s in pdf_content.sections] if pdf_content else None,
                references=[{"text": ref} for ref in pdf_content.references] if pdf_content else None,
                parser_used=pdf_content.parser_used.value if pdf_content else None,
                parser_metadata=pdf_content.metadata if pdf_content else None,
                pdf_processed=pdf_content is not None,
                pdf_processing_date=datetime.now() if pdf_content else None,
                content_hash=content_hash,
            )
            repo.upsert(paper_create)
            session.commit()

            logger.info(f"Ingested paper {arxiv_id}: {len(chunks) if chunks else 0} chunks indexed")
    except Exception as e:
        logger.error(f"Ingest failed for {arxiv_id}: {e}")


async def _bulk_ingest_worker(
    task_id: str,
    arxiv_ids: list[str],
    opensearch_client,
    embeddings_service,
):
    """Background worker that ingests a list of arxiv IDs sequentially.

    Uses a single shared database connection for the entire batch.
    Updates task status as papers are processed.
    """
    from src.database import get_database

    task = _task_store.get(task_id)
    database = get_database()
    try:
        for arxiv_id in arxiv_ids:
            try:
                await _ingest_single_paper(
                    arxiv_id=arxiv_id,
                    opensearch_client=opensearch_client,
                    embeddings_service=embeddings_service,
                    database=database,
                )
                if task:
                    task.completed += 1
            except Exception as e:
                logger.error(f"Bulk ingest error for {arxiv_id}: {e}")
                if task:
                    task.failed += 1
                    task.errors.append(f"{arxiv_id}: {e}")
    finally:
        if task:
            task.status = "completed" if task.failed == 0 else "completed_with_errors"
            task.finished_at = time.time()


@router.post("/bulk-import", response_model=BulkImportResponse)
async def bulk_import_papers(
    request: BulkImportRequest,
    background_tasks: BackgroundTasks,
    opensearch_client: OpenSearchDep,
    embeddings_service: EmbeddingsDep,
    _key: APIKeyDep,
):
    """
    Bulk import papers from arXiv.

    Accepts either a list of arXiv IDs or a category to fetch recent papers from.
    Returns a task_id for tracking progress.
    """
    arxiv_ids = request.arxiv_ids

    if request.category and not arxiv_ids:
        try:
            from src.services.arxiv.factory import make_arxiv_client

            arxiv_client = make_arxiv_client()
            # Override category on a local copy, not the shared client's settings
            original = arxiv_client._settings.model_copy()
            arxiv_client._settings.search_category = request.category
            papers = await arxiv_client.fetch_papers(max_results=50)
            arxiv_ids = [p.arxiv_id for p in papers]
            arxiv_client._settings = original
        except Exception as e:
            logger.error(f"Failed to fetch papers for category {request.category}: {e}")
            raise HTTPException(
                status_code=502,
                detail=f"Failed to fetch papers from category '{request.category}'",
            )

    if not arxiv_ids:
        raise HTTPException(
            status_code=422,
            detail="Either arxiv_ids or category must result in at least one paper",
        )

    task_id = str(uuid.uuid4())

    # Register task in the store before dispatching
    _task_store[task_id] = TaskStatus(task_id=task_id, total=len(arxiv_ids))

    background_tasks.add_task(
        _bulk_ingest_worker,
        task_id=task_id,
        arxiv_ids=arxiv_ids,
        opensearch_client=opensearch_client,
        embeddings_service=embeddings_service,
    )

    logger.info(f"Bulk import started: task_id={task_id}, papers={len(arxiv_ids)}")

    return BulkImportResponse(
        task_id=task_id,
        total_submitted=len(arxiv_ids),
        status="processing",
    )


@router.get("/bulk-import/{task_id}/status")
async def get_bulk_import_status(
    task_id: str,
    _key: APIKeyDep,
):
    """Check the status of a bulk import task."""
    task = _get_task_status(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    elapsed = (task.finished_at or time.time()) - task.started_at
    return {
        "task_id": task.task_id,
        "status": task.status,
        "total": task.total,
        "completed": task.completed,
        "failed": task.failed,
        "elapsed_seconds": round(elapsed, 2),
        "errors": task.errors[:10],
    }
