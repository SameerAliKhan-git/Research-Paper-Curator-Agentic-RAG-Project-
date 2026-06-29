#!/usr/bin/env python3
import asyncio
import logging
import sys
from pathlib import Path

# Setup path so script runs correctly
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.database import init_database
from src.repositories.paper import PaperRepository
from src.services.opensearch.factory import make_opensearch_client
from src.services.embeddings.factory import make_embeddings_service
from src.services.vision.colpali import ColPaliVisionService

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("prerender_pages")

async def prerender_and_index():
    """Find all indexed papers and generate visual page layout screenshots & embeddings."""
    logger.info("Initializing services...")
    db = init_database()
    opensearch_client = make_opensearch_client()
    embeddings_service = make_embeddings_service()
    
    colpali_service = ColPaliVisionService(opensearch_client, embeddings_service)
    
    # Create session
    with db.get_session() as session:
        repo = PaperRepository(session)
        papers = repo.get_all(limit=100)
        
        if not papers:
            logger.warning("No papers found in database. Please index some papers first.")
            return

        logger.info(f"Found {len(papers)} papers in database to process.")
        
        for paper in papers:
            logger.info(f"Processing paper {paper.arxiv_id}: {paper.title[:50]}...")
            
            # Simulate 3 pages per paper for the visual demo
            pages_to_render = 3
            
            for page_num in range(1, pages_to_render + 1):
                # Layout distribution depending on page index
                if page_num == 1:
                    paragraphs = 5
                    tables = 1
                    pictures = 1
                    equations = 1
                elif page_num == 2:
                    paragraphs = 8
                    tables = 0
                    pictures = 2
                    equations = 2
                else:
                    paragraphs = 6
                    tables = 2
                    pictures = 0
                    equations = 1
                
                # Mock some page text
                page_text = f"This is the parsed OCR text of page {page_num} of the paper '{paper.title}' with arXiv ID {paper.arxiv_id}. "
                if page_num == 1:
                    page_text += f"Abstract: {paper.abstract[:200]}"
                elif page_num == 2:
                    page_text += "Methodology: We propose an agentic workflow using dense vector embeddings and open-source models."
                else:
                    page_text += f"Results: Our approach achieves state of the art results with {tables} comparative tables."

                # Index page details (index_visual_page generates visual layout screenshot internally)
                await colpali_service.index_visual_page(
                    arxiv_id=paper.arxiv_id,
                    paper_id=str(paper.id),
                    page_number=page_num,
                    page_text=page_text,
                    layout_stats={
                        "text_regions": paragraphs,
                        "tables": tables,
                        "pictures": pictures,
                        "equations": equations
                    },
                    tenant_id="default"
                )
                logger.info(f"  Indexed page {page_num}/{pages_to_render} successfully.")

    logger.info("All visual pages have been pre-rendered and indexed successfully!")

if __name__ == "__main__":
    asyncio.run(prerender_and_index())
