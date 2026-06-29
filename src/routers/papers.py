import hashlib
import logging
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from src.dependencies import APIKeyDep, EmbeddingsDep, LangfuseDep, OpenSearchDep, SessionDep, SettingsDep
from src.repositories.paper import PaperRepository
from src.services.paper_sync import PaperSyncService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/papers", tags=["papers"])


@router.get("/")
def list_papers(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: SessionDep = None,
    _key: APIKeyDep = None,
):
    """
    List all indexed papers from PostgreSQL database.
    """
    repo = PaperRepository(db)
    papers = repo.get_all(limit=limit, offset=offset)
    total = repo.get_count()

    results = []
    for paper in papers:
        authors_val = paper.authors
        if isinstance(authors_val, str):
            authors_val = [authors_val]

        results.append(
            {
                "id": str(paper.id),
                "arxiv_id": paper.arxiv_id,
                "title": paper.title,
                "authors": authors_val,
                "abstract": paper.abstract,
                "categories": paper.categories if isinstance(paper.categories, list) else [paper.categories],
                "published_date": paper.published_date.isoformat() if paper.published_date else None,
                "pdf_url": paper.pdf_url,
                "pdf_processed": paper.pdf_processed,
            }
        )
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_more": (offset + limit) < total,
        "papers": results,
    }


@router.post("/sync")
async def sync_papers(
    db: SessionDep,
    opensearch_client: OpenSearchDep,
    _key: APIKeyDep,
    request: Request,
):
    """Sync papers from OpenSearch to PostgreSQL."""
    try:
        sync_service = PaperSyncService(opensearch_client=opensearch_client, db_session=db)
        result = sync_service.sync_all()

        # Clear exact and semantic caches
        cache_client = getattr(request.app.state, "cache_client", None)
        semantic_cache = getattr(request.app.state, "semantic_cache", None)
        if cache_client:
            await cache_client.clear()
        if semantic_cache:
            await semantic_cache.clear()

        return result
    except Exception as e:
        logger.error("Sync failed")
        raise HTTPException(status_code=500, detail="Sync operation failed")


@router.post("/ingest")
async def ingest_paper(
    arxiv_id: str,
    background_tasks: BackgroundTasks,
    opensearch_client: OpenSearchDep,
    embeddings_service: EmbeddingsDep,
    langfuse_tracer: LangfuseDep,
    db: SessionDep,
    _key: APIKeyDep,
    request: Request,
):
    """Ingest a single paper from arXiv: fetch, parse, chunk, index to OpenSearch + PostgreSQL."""
    from src.database import get_database
    from src.services.arxiv.factory import make_arxiv_client
    from src.services.pdf_parser.factory import make_pdf_parser_service

    async def _do_ingest(arxiv_id: str):
        import uuid
        from datetime import datetime

        from src.schemas.arxiv.paper import PaperCreate
        from src.services.indexing.factory import make_hybrid_indexing_service

        database = get_database()
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
                        indexer = make_hybrid_indexing_service()
                        paper_id = existing.id if existing else uuid.uuid4()

                        # Parse published_date string to datetime
                        pub_date_str = paper.published_date
                        try:
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

                            logger.info(f"Processing figures for paper {arxiv_id}...")
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

                logger.info(f"Ingested paper {arxiv_id} successfully")

                # Invalidate/clear semantic and exact cache selectively
                cache_client = getattr(request.app.state, "cache_client", None)
                semantic_cache = getattr(request.app.state, "semantic_cache", None)
                if semantic_cache and paper.abstract:
                    try:
                        paper_embedding = await embeddings_service.embed_query(paper.abstract)
                        await semantic_cache.invalidate_similar(paper_embedding, threshold=0.70)
                    except Exception as cache_err:
                        logger.error(f"Selective cache invalidation failed: {cache_err}. Falling back to full clear.")
                        if cache_client:
                            await cache_client.clear()
                        await semantic_cache.clear()
                else:
                    if cache_client:
                        await cache_client.clear()
                    if semantic_cache:
                        await semantic_cache.clear()
        except Exception as e:
            logger.error(f"Ingest failed for {arxiv_id}: {e}")

    background_tasks.add_task(_do_ingest, arxiv_id)
    return {"status": "ingestion_started", "arxiv_id": arxiv_id}


@router.get("/stats")
async def paper_stats(
    opensearch_client: OpenSearchDep,
    db: SessionDep,
    _key: APIKeyDep = None,
):
    """Get paper statistics from both OpenSearch and PostgreSQL."""
    try:
        # OpenSearch stats
        os_count = 0
        try:
            result = opensearch_client.client.count(index=opensearch_client.index_name)
            os_count = result.get("count", 0)
        except Exception:
            pass

        # PostgreSQL stats - use efficient count query
        repo = PaperRepository(db)
        pg_count = repo.get_count()

        return {
            "opensearch_chunks": os_count,
            "postgresql_papers": pg_count,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")


@router.get("/{arxiv_id}/bibtex")
def get_paper_bibtex(
    arxiv_id: str,
    db: SessionDep,
    _key: APIKeyDep = None,
):
    """
    Generate a BibTeX citation entry for the requested paper.
    """
    import json

    repo = PaperRepository(db)
    paper = repo.get_by_arxiv_id(arxiv_id)
    if not paper:
        raise HTTPException(status_code=404, detail=f"Paper {arxiv_id} not found")

    authors_val = paper.authors
    if isinstance(authors_val, str):
        try:
            parsed = json.loads(authors_val)
            if isinstance(parsed, list):
                authors_list = parsed
            else:
                authors_list = [str(parsed)]
        except Exception:
            authors_list = [authors_val]
    elif isinstance(authors_val, list):
        authors_list = authors_val
    else:
        authors_list = []

    authors_str = " and ".join(authors_list) if authors_list else "Unknown"
    year = paper.published_date.year if paper.published_date else 2026
    bib_key = f"arxiv_{arxiv_id.replace('.', '_').replace('/', '_')}"

    bibtex = f"""@article{{{bib_key},
  author    = {{{authors_str}}},
  title     = {{{paper.title}}},
  journal   = {{arXiv preprint arXiv:{arxiv_id}}},
  year      = {{{year}}},
  url       = {{{paper.pdf_url or f"https://arxiv.org/abs/{arxiv_id}"}}}
}}"""

    return {"arxiv_id": arxiv_id, "bibtex": bibtex}


@router.get("/check/{arxiv_id}")
def check_paper_exists(
    arxiv_id: str,
    db: SessionDep,
    _key: APIKeyDep = None,
):
    """
    Check if a paper with the given arXiv ID has been fully ingested.
    """
    repo = PaperRepository(db)
    paper = repo.get_by_arxiv_id(arxiv_id)
    return {"exists": paper is not None, "title": paper.title if paper else None}
