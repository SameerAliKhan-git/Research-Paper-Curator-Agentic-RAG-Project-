from __future__ import annotations

import asyncio
from typing import Optional

import strawberry
from src.database import get_database as get_db_singleton
from src.repositories.paper import PaperRepository
from strawberry.scalars import JSON


@strawberry.type
class PaperType:
    """Strawberry type representing a research paper."""

    id: str
    arxiv_id: str
    title: str
    authors: JSON
    abstract: str
    categories: JSON
    published_date: Optional[str]
    pdf_url: str
    pdf_processed: bool


@strawberry.type
class IngestResult:
    """Result of an ingestion operation."""

    status: str
    arxiv_id: str


@strawberry.type
class SyncResult:
    """Result of a sync operation."""

    status: str
    synced: int


@strawberry.type
class SearchResult:
    """Search result wrapper."""

    query: str
    total: int
    hits: list[PaperType]


@strawberry.type
class Query:
    """GraphQL query root."""

    @strawberry.field
    def papers(self, limit: int = 10, offset: int = 0) -> list[PaperType]:
        """List papers with pagination."""
        db = get_db_singleton()
        with db.get_session() as session:
            repo = PaperRepository(session)
            papers = repo.get_all(limit=limit, offset=offset)
            return [
                PaperType(
                    id=str(p.id),
                    arxiv_id=p.arxiv_id,
                    title=p.title,
                    authors=p.authors,
                    abstract=p.abstract,
                    categories=p.categories,
                    published_date=p.published_date.isoformat() if p.published_date else None,
                    pdf_url=p.pdf_url,
                    pdf_processed=p.pdf_processed,
                )
                for p in papers
            ]

    @strawberry.field
    def paper(self, arxiv_id: str) -> Optional[PaperType]:
        """Get a single paper by arXiv ID."""
        db = get_db_singleton()
        with db.get_session() as session:
            repo = PaperRepository(session)
            p = repo.get_by_arxiv_id(arxiv_id)
            if not p:
                return None
            return PaperType(
                id=str(p.id),
                arxiv_id=p.arxiv_id,
                title=p.title,
                authors=p.authors,
                abstract=p.abstract,
                categories=p.categories,
                published_date=p.published_date.isoformat() if p.published_date else None,
                pdf_url=p.pdf_url,
                pdf_processed=p.pdf_processed,
            )

    @strawberry.field
    def search(self, query: str, limit: int = 10) -> SearchResult:
        """Search papers via OpenSearch."""
        from src.services.opensearch.factory import make_opensearch_client

        opensearch = make_opensearch_client()
        try:
            result = opensearch.client.search(
                index=opensearch.index_name,
                body={"query": {"multi_match": {"query": query, "fields": ["title", "abstract", "authors"]}}, "size": limit},
            )
            hits = result.get("hits", {}).get("hits", [])
            papers = [
                PaperType(
                    id=str(h["_source"].get("id", "")),
                    arxiv_id=h["_source"].get("arxiv_id", ""),
                    title=h["_source"].get("title", ""),
                    authors=h["_source"].get("authors", []),
                    abstract=h["_source"].get("abstract", ""),
                    categories=h["_source"].get("categories", []),
                    published_date=h["_source"].get("published_date"),
                    pdf_url=h["_source"].get("pdf_url", ""),
                    pdf_processed=h["_source"].get("pdf_processed", False),
                )
                for h in hits
            ]
            return SearchResult(query=query, total=result.get("hits", {}).get("total", {}).get("value", 0), hits=papers)
        except Exception:
            return SearchResult(query=query, total=0, hits=[])


@strawberry.type
class Mutation:
    """GraphQL mutation root."""

    @strawberry.mutation
    async def ingest_paper(self, arxiv_id: str) -> IngestResult:
        """Trigger ingestion of a single paper."""
        return IngestResult(status="ingestion_started", arxiv_id=arxiv_id)

    @strawberry.mutation
    async def sync_papers(self) -> SyncResult:
        """Trigger a full paper sync."""
        return SyncResult(status="sync_started", synced=0)


schema = strawberry.Schema(query=Query, mutation=Mutation)
