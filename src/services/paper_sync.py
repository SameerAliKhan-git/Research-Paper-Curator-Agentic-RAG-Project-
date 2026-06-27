"""Service to sync papers from OpenSearch to PostgreSQL."""

import logging
from typing import Dict, List, Optional

from sqlalchemy.orm import Session
from src.repositories.paper import PaperRepository
from src.schemas.arxiv.paper import PaperCreate
from src.services.opensearch.client import OpenSearchClient

logger = logging.getLogger(__name__)


class PaperSyncService:
    """Syncs paper metadata from OpenSearch index into PostgreSQL."""

    def __init__(self, opensearch_client: OpenSearchClient, db_session: Session):
        self.opensearch = opensearch_client
        self.session = db_session
        self.repo = PaperRepository(db_session)

    def sync_all(self, batch_size: int = 100, max_offset: int = 100000) -> Dict[str, int]:
        """Scan OpenSearch for unique papers and upsert each into PostgreSQL.

        Uses a raw match_all query to retrieve all documents, since
        search_unified's BM25 mode doesn't support wildcard queries.

        Args:
            batch_size: Number of documents to fetch per batch
            max_offset: Maximum offset to prevent infinite loops

        Returns counts of synced and failed papers.
        """
        synced = 0
        failed = 0
        seen_ids: set = set()
        offset = 0

        while offset < max_offset:
            try:
                # Use raw OpenSearch match_all query
                body = {
                    "query": {"match_all": {}},
                    "size": batch_size,
                    "from": offset,
                    "_source": ["arxiv_id", "title", "authors", "abstract", "categories", "published_date", "pdf_url"],
                }
                resp = self.opensearch.client.search(
                    index=self.opensearch.index_name,
                    body=body,
                )
                hits = resp.get("hits", {}).get("hits", [])
            except Exception as e:
                logger.warning(f"OpenSearch query failed at offset {offset}: {e}")
                break
            if not hits:
                break

            for hit in hits:
                source = hit.get("_source", {})
                arxiv_id = source.get("arxiv_id", "")
                if not arxiv_id or arxiv_id in seen_ids:
                    continue
                seen_ids.add(arxiv_id)

                try:
                    paper_create = PaperCreate(
                        arxiv_id=arxiv_id,
                        title=source.get("title", "Untitled"),
                        authors=source.get("authors", []) if isinstance(source.get("authors"), list) else [],
                        abstract=source.get("abstract", source.get("chunk_text", "")),
                        categories=source.get("categories", []) if isinstance(source.get("categories"), list) else [],
                        published_date=source.get("published_date"),
                        pdf_url=source.get("pdf_url")
                        or (f"https://arxiv.org/pdf/{arxiv_id}.pdf" if not arxiv_id.startswith("upload_") else "#"),
                    )
                    self.repo.upsert(paper_create)
                    synced += 1
                except Exception as e:
                    logger.warning(f"Failed to sync paper {arxiv_id}: {e}")
                    failed += 1

            offset += batch_size

        self.session.commit()
        logger.info(f"Sync complete: {synced} synced, {failed} failed, {len(seen_ids)} total")
        return {"synced": synced, "failed": failed, "total": len(seen_ids)}

    def sync_single(self, arxiv_id: str) -> Optional[str]:
        """Sync a single paper from OpenSearch to PostgreSQL.

        Returns the arxiv_id on success, None on failure.
        """
        try:
            body = {
                "query": {"term": {"arxiv_id": arxiv_id}},
                "size": 1,
                "_source": ["arxiv_id", "title", "authors", "abstract", "categories", "published_date", "pdf_url"],
            }
            resp = self.opensearch.client.search(
                index=self.opensearch.index_name,
                body=body,
            )
            hits = resp.get("hits", {}).get("hits", [])
        except Exception as e:
            logger.warning(f"OpenSearch query failed for {arxiv_id}: {e}")
            return None

        if not hits:
            return None

        source = hits[0].get("_source", {})
        try:
            paper_create = PaperCreate(
                arxiv_id=source.get("arxiv_id", arxiv_id),
                title=source.get("title", "Untitled"),
                authors=source.get("authors", []) if isinstance(source.get("authors"), list) else [],
                abstract=source.get("abstract", source.get("chunk_text", "")),
                categories=source.get("categories", []) if isinstance(source.get("categories"), list) else [],
                published_date=source.get("published_date"),
                pdf_url=source.get("pdf_url")
                or (f"https://arxiv.org/pdf/{arxiv_id}.pdf" if not arxiv_id.startswith("upload_") else "#"),
            )
            self.repo.upsert(paper_create)
            self.session.commit()
            return arxiv_id
        except Exception as e:
            logger.warning(f"Failed to sync paper {arxiv_id}: {e}")
            self.session.rollback()
            return None
