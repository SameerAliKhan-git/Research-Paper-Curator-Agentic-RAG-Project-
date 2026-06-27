"""Script to fetch arXiv papers, parse PDFs, and index them in OpenSearch.

Usage:
    python -m src.scripts.arxiv_ingest --query "cat:cs.AI AND ti:transformer" --max-papers 10
    python -m src.scripts.arxiv_ingest --query "cat:cs.CL AND abs:language model" --max-papers 5

Prerequisites:
    - OpenSearch running and accessible
    - PostgreSQL running and accessible
    - Jina API key configured for embeddings
    - Docling installed for PDF parsing
"""

import argparse
import asyncio
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import get_settings
from src.db.factory import make_database
from src.models.paper import Paper
from src.repositories.paper import PaperRepository
from src.schemas.arxiv.paper import PaperCreate
from src.services.arxiv.client import ArxivClient
from src.services.embeddings.factory import make_embeddings_service
from src.services.indexing.text_chunker import TextChunker
from src.services.opensearch.factory import make_opensearch_client
from src.services.pdf_parser.factory import make_pdf_parser_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ArxivIngester:
    """Orchestrates the full ingestion pipeline: fetch -> download -> parse -> chunk -> embed -> index."""

    def __init__(self, tenant_id: str = "default"):
        self.settings = get_settings()
        self.tenant_id = tenant_id
        self.arxiv_client = ArxivClient(self.settings.arxiv)
        self.pdf_parser = make_pdf_parser_service()
        self.embeddings_client = make_embeddings_service()
        self.opensearch_client = make_opensearch_client()
        self.text_chunker = TextChunker(
            chunk_size=self.settings.chunking.chunk_size,
            overlap_size=self.settings.chunking.overlap_size,
            min_chunk_size=self.settings.chunking.min_chunk_size,
        )
        self.database = make_database()

    async def ingest(
        self,
        query: str,
        max_papers: int = 10,
        skip_existing: bool = True,
        force_download: bool = False,
    ) -> dict:
        """Run the full ingestion pipeline.

        :param query: arXiv search query (e.g., "cat:cs.AI AND ti:transformer")
        :param max_papers: Maximum number of papers to ingest
        :param skip_existing: Skip papers already indexed
        :param force_download: Force re-download of PDFs
        :returns: Ingestion statistics
        """
        stats = {
            "total_fetched": 0,
            "papers_ingested": 0,
            "papers_skipped": 0,
            "papers_failed": 0,
            "total_chunks": 0,
            "errors": [],
        }

        logger.info(f"Starting ingestion: query='{query}', max_papers={max_papers}")

        try:
            # Step 1: Fetch papers from arXiv
            logger.info("Step 1: Fetching papers from arXiv API...")
            papers = await self.arxiv_client.fetch_papers_with_query(
                search_query=query,
                max_results=max_papers,
            )
            stats["total_fetched"] = len(papers)
            logger.info(f"Fetched {len(papers)} papers from arXiv")

            if not papers:
                logger.warning("No papers found for the given query")
                return stats

            # Step 2: Process each paper
            for i, paper in enumerate(papers, 1):
                logger.info(f"\n--- Paper {i}/{len(papers)}: {paper.arxiv_id} ---")
                try:
                    # Check if already exists
                    if skip_existing:
                        if self._paper_exists(paper.arxiv_id):
                            logger.info(f"Skipping {paper.arxiv_id} (already indexed)")
                            stats["papers_skipped"] += 1
                            continue

                    # Download and parse PDF
                    pdf_path = await self.arxiv_client.download_pdf(paper, force_download=force_download)
                    if not pdf_path:
                        logger.warning(f"Failed to download PDF for {paper.arxiv_id}")
                        stats["papers_failed"] += 1
                        stats["errors"].append(f"PDF download failed: {paper.arxiv_id}")
                        continue

                    # Parse PDF
                    pdf_content = await self.pdf_parser.parse_pdf(pdf_path)
                    if not pdf_content:
                        logger.warning(f"Failed to parse PDF for {paper.arxiv_id}")
                        stats["papers_failed"] += 1
                        stats["errors"].append(f"PDF parse failed: {paper.arxiv_id}")
                        continue

                    # Store paper metadata in PostgreSQL
                    paper_id = self._store_paper_metadata(paper, pdf_content)
                    if not paper_id:
                        logger.warning(f"Failed to store metadata for {paper.arxiv_id}")
                        stats["papers_failed"] += 1
                        stats["errors"].append(f"Metadata store failed: {paper.arxiv_id}")
                        continue

                    # Chunk the text
                    full_text = pdf_content.full_text or ""
                    sections = None
                    if hasattr(pdf_content, "sections") and pdf_content.sections:
                        sections = pdf_content.sections

                    chunks = self.text_chunker.chunk_paper(
                        title=paper.title,
                        abstract=paper.abstract,
                        full_text=full_text,
                        arxiv_id=paper.arxiv_id,
                        paper_id=str(paper_id),
                        sections=sections,
                    )

                    if not chunks:
                        logger.warning(f"No chunks created for {paper.arxiv_id}")
                        stats["papers_failed"] += 1
                        stats["errors"].append(f"No chunks: {paper.arxiv_id}")
                        continue

                    logger.info(f"Created {len(chunks)} chunks for {paper.arxiv_id}")

                    # Embed chunks
                    chunk_texts = [chunk.text for chunk in chunks]
                    embeddings = await self.embeddings_client.embed_passages(chunk_texts)

                    if len(embeddings) != len(chunks):
                        logger.warning(f"Embedding count mismatch for {paper.arxiv_id}")
                        stats["papers_failed"] += 1
                        stats["errors"].append(f"Embedding mismatch: {paper.arxiv_id}")
                        continue

                    # Index in OpenSearch
                    indexed = self._index_chunks(paper, chunks, embeddings)
                    if indexed:
                        stats["papers_ingested"] += 1
                        stats["total_chunks"] += len(chunks)
                        logger.info(f"Successfully ingested {paper.arxiv_id} ({len(chunks)} chunks)")

                        # Process and index multimodal figures
                        try:
                            from src.services.multimodal import MultiModalProcessor
                            from src.services.ollama.factory import make_ollama_client

                            ollama_client = make_ollama_client()
                            multimodal_processor = MultiModalProcessor()

                            logger.info(f"Processing figures for ingested paper {paper.arxiv_id}...")
                            await multimodal_processor.process_and_index_pdf_figures(
                                pdf_path=pdf_path,
                                paper_id=str(paper_id),
                                opensearch_client=self.opensearch_client,
                                ollama_client=ollama_client,
                                embeddings_client=self.embeddings_client,
                                tenant_id=self.tenant_id,
                            )
                        except Exception as mm_err:
                            logger.warning(f"Failed to process multimodal figures for {paper.arxiv_id}: {mm_err}")
                    else:
                        stats["papers_failed"] += 1
                        stats["errors"].append(f"Indexing failed: {paper.arxiv_id}")

                except Exception as e:
                    logger.error(f"Error processing {paper.arxiv_id}: {e}")
                    stats["papers_failed"] += 1
                    stats["errors"].append(f"Exception: {paper.arxiv_id} - {str(e)}")

            logger.info(f"\n{'=' * 60}")
            logger.info("Ingestion complete!")
            logger.info(f"  Papers fetched: {stats['total_fetched']}")
            logger.info(f"  Papers ingested: {stats['papers_ingested']}")
            logger.info(f"  Papers skipped: {stats['papers_skipped']}")
            logger.info(f"  Papers failed: {stats['papers_failed']}")
            logger.info(f"  Total chunks: {stats['total_chunks']}")
            if stats["errors"]:
                logger.info(f"  Errors: {len(stats['errors'])}")
                for err in stats["errors"][:5]:  # Show first 5 errors
                    logger.info(f"    - {err}")
            logger.info(f"{'=' * 60}")

            return stats

        except Exception as e:
            logger.error(f"Ingestion pipeline failed: {e}")
            raise
        finally:
            self.database.teardown()

    def _paper_exists(self, arxiv_id: str) -> bool:
        """Check if paper already exists in the database."""
        try:
            with self.database.get_session() as session:
                repo = PaperRepository(session)
                existing = repo.get_by_arxiv_id(arxiv_id)
                return existing is not None and existing.pdf_processed
        except Exception as e:
            logger.debug(f"Error checking paper existence: {e}")
            return False

    def _store_paper_metadata(self, arxiv_paper, pdf_content) -> str | None:
        """Store paper metadata and parsed content in PostgreSQL."""
        try:
            with self.database.get_session() as session:
                repo = PaperRepository(session)

                # Check if paper exists
                existing = repo.get_by_arxiv_id(arxiv_paper.arxiv_id)

                if existing:
                    # Update existing paper
                    existing.raw_text = pdf_content.full_text or ""
                    existing.sections = [s.model_dump() for s in pdf_content.sections] if hasattr(pdf_content, "sections") and pdf_content.sections else None
                    existing.references = [{"text": r} for r in pdf_content.references] if hasattr(pdf_content, "references") and pdf_content.references else None
                    existing.parser_used = "docling"
                    existing.pdf_processed = True
                    existing.pdf_processing_date = datetime.now(timezone.utc)
                    repo.update(existing)
                    return existing.id
                else:
                    # Create new paper
                    paper_create = PaperCreate(
                        arxiv_id=arxiv_paper.arxiv_id,
                        title=arxiv_paper.title,
                        authors=arxiv_paper.authors,
                        abstract=arxiv_paper.abstract,
                        categories=arxiv_paper.categories,
                        published_date=datetime.fromisoformat(arxiv_paper.published_date.replace("Z", "+00:00")),
                        pdf_url=arxiv_paper.pdf_url,
                        raw_text=pdf_content.full_text or "",
                        sections=[s.model_dump() for s in pdf_content.sections] if hasattr(pdf_content, "sections") and pdf_content.sections else None,
                        references=[{"text": r} for r in pdf_content.references] if hasattr(pdf_content, "references") and pdf_content.references else None,
                        parser_used="docling",
                        pdf_processed=True,
                        pdf_processing_date=datetime.now(timezone.utc),
                    )
                    paper = repo.create(paper_create)
                    return paper.id

        except Exception as e:
            logger.error(f"Error storing paper metadata: {e}")
            return None

    def _index_chunks(self, arxiv_paper, chunks, embeddings) -> bool:
        """Index chunks with embeddings in OpenSearch."""
        try:
            bulk_chunks = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_data = {
                    "arxiv_id": arxiv_paper.arxiv_id,
                    "tenant_id": self.tenant_id,
                    "chunk_index": chunk.metadata.chunk_index,
                    "chunk_text": chunk.text,
                    "chunk_word_count": chunk.metadata.word_count,
                    "start_char": chunk.metadata.start_char,
                    "end_char": chunk.metadata.end_char,
                    "title": arxiv_paper.title,
                    "authors": arxiv_paper.authors,
                    "abstract": arxiv_paper.abstract,
                    "categories": arxiv_paper.categories,
                    "published_date": arxiv_paper.published_date,
                    "section_title": chunk.metadata.section_title or "",
                    "embedding_model": "jina-embeddings-v3",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                bulk_chunks.append(
                    {
                        "chunk_data": chunk_data,
                        "embedding": embedding,
                    }
                )

            result = self.opensearch_client.bulk_index_chunks(bulk_chunks)
            logger.info(f"Indexed {result['success']} chunks for {arxiv_paper.arxiv_id}")
            return result["success"] > 0

        except Exception as e:
            logger.error(f"Error indexing chunks: {e}")
            return False


async def main():
    parser = argparse.ArgumentParser(description="Ingest arXiv papers into the RAG system")
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help='arXiv search query (e.g., "cat:cs.AI AND ti:transformer")',
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=10,
        help="Maximum number of papers to ingest (default: 10)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip papers already indexed (default: True)",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        default=False,
        help="Force re-download of PDFs (default: False)",
    )

    args = parser.parse_args()

    ingester = ArxivIngester()
    stats = await ingester.ingest(
        query=args.query,
        max_papers=args.max_papers,
        skip_existing=args.skip_existing,
        force_download=args.force_download,
    )

    # Exit with error if all papers failed
    if stats["papers_failed"] == stats["total_fetched"] and stats["total_fetched"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
