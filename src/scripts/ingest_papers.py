"""Quick ingestion of specific arXiv papers by ID.

Usage:
    python -m src.scripts.ingest_papers --ids 1810.04805 1706.03762 2005.14165
    python -m src.scripts.ingest_papers --ids 1810.04805 --query "attention mechanism"

This script is useful for testing the RAG system with specific well-known papers.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def ingest_by_ids(arxiv_ids: list[str], tenant_id: str = "default"):
    """Ingest specific papers by their arXiv IDs."""
    from src.scripts.arxiv_ingest import ArxivIngester

    ingester = ArxivIngester(tenant_id=tenant_id)

    for arxiv_id in arxiv_ids:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Ingesting paper: {arxiv_id}")
        logger.info(f"{'=' * 60}")

        try:
            # Fetch paper metadata
            paper = await ingester.arxiv_client.fetch_paper_by_id(arxiv_id)
            if not paper:
                logger.error(f"Paper {arxiv_id} not found on arXiv")
                continue

            logger.info(f"Title: {paper.title}")
            logger.info(f"Authors: {', '.join(paper.authors[:3])}...")

            # Check if already exists
            if ingester._paper_exists(arxiv_id):
                logger.info(f"Paper {arxiv_id} already indexed, skipping")
                continue

            # Download PDF
            pdf_path = await ingester.arxiv_client.download_pdf(paper, force_download=False)
            if not pdf_path:
                logger.error(f"Failed to download PDF for {arxiv_id}")
                continue

            # Parse PDF
            pdf_content = await ingester.pdf_parser.parse_pdf(pdf_path)
            if not pdf_content:
                logger.error(f"Failed to parse PDF for {arxiv_id}")
                continue

            # Store metadata
            paper_id = ingester._store_paper_metadata(paper, pdf_content)
            if not paper_id:
                logger.error(f"Failed to store metadata for {arxiv_id}")
                continue

            # Chunk text
            full_text = pdf_content.full_text or ""
            sections = pdf_content.sections if hasattr(pdf_content, "sections") else None

            chunks = ingester.text_chunker.chunk_paper(
                title=paper.title,
                abstract=paper.abstract,
                full_text=full_text,
                arxiv_id=arxiv_id,
                paper_id=str(paper_id),
                sections=sections,
            )

            if not chunks:
                logger.error(f"No chunks created for {arxiv_id}")
                continue

            # Embed chunks
            chunk_texts = [chunk.text for chunk in chunks]
            embeddings = await ingester.embeddings_client.embed_passages(chunk_texts)

            # Index in OpenSearch
            success = ingester._index_chunks(paper, chunks, embeddings)
            if success:
                logger.info(f"Successfully ingested {arxiv_id} ({len(chunks)} chunks)")
            else:
                logger.error(f"Failed to index chunks for {arxiv_id}")

        except Exception as e:
            logger.error(f"Error ingesting {arxiv_id}: {e}")
            import traceback

            traceback.print_exc()

    ingester.database.teardown()


async def ingest_bert_papers(tenant_id: str = "default"):
    """Ingest the most important BERT-related papers for testing."""
    bert_paper_ids = [
        "1810.04805",  # BERT: Pre-training of Deep Bidirectional Transformers
        "1706.03762",  # Attention Is All You Need (Transformer)
        "2005.14165",  # GPT-3
        "1905.10044",  # BERTScore
        "1802.05365",  # BERT for Reading Comprehension
    ]

    logger.info("Ingesting key BERT/Transformer papers...")
    await ingest_by_ids(bert_paper_ids, tenant_id=tenant_id)


def main():
    parser = argparse.ArgumentParser(description="Ingest specific arXiv papers by ID")
    parser.add_argument(
        "--ids",
        nargs="+",
        help="List of arXiv paper IDs to ingest (e.g., 1810.04805 1706.03762)",
    )
    parser.add_argument(
        "--bert-papers",
        action="store_true",
        help="Ingest the 5 most important BERT/Transformer papers",
    )
    parser.add_argument(
        "--tenant-id",
        type=str,
        default="default",
        help="Tenant ID for index isolation (default: default)",
    )

    args = parser.parse_args()

    if args.bert_papers:
        asyncio.run(ingest_bert_papers(tenant_id=args.tenant_id))
    elif args.ids:
        asyncio.run(ingest_by_ids(args.ids, tenant_id=args.tenant_id))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
