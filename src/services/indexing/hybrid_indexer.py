import logging
from typing import Dict, List, Optional

from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.opensearch.client import OpenSearchClient

from .text_chunker import TextChunker

logger = logging.getLogger(__name__)


class HybridIndexingService:
    """Service for indexing papers with chunking and embeddings for hybrid search.

    Orchestrates the process of:
    1. Chunking papers into overlapping segments
    2. Generating embeddings for each chunk
    3. Indexing chunks with embeddings into OpenSearch
    """

    def __init__(self, chunker: TextChunker, embeddings_client: JinaEmbeddingsClient, opensearch_client: OpenSearchClient):
        """Initialize hybrid indexing service.

        :param chunker: Text chunking service
        :param embeddings_client: Embeddings generation client
        :param opensearch_client: OpenSearch client
        """
        self.chunker = chunker
        self.embeddings_client = embeddings_client
        self.opensearch_client = opensearch_client

        logger.info("Hybrid indexing service initialized")

    async def index_paper(self, paper_data: Dict) -> Dict[str, int]:
        """Index a single paper with chunking and embeddings.

        :param paper_data: Paper data from database
        :returns: Dictionary with indexing statistics
        """
        arxiv_id = paper_data.get("arxiv_id")
        paper_id = str(paper_data.get("id", ""))
        tenant_id = paper_data.get("tenant_id")

        if not arxiv_id:
            logger.error("Paper missing arxiv_id")
            return {"chunks_created": 0, "chunks_indexed": 0, "embeddings_generated": 0, "errors": 1}

        try:
            # Step 1: Chunk the paper using hybrid section-based approach
            chunks = self.chunker.chunk_paper(
                title=paper_data.get("title", ""),
                abstract=paper_data.get("abstract", ""),
                full_text=paper_data.get("raw_text", paper_data.get("full_text", "")),
                arxiv_id=arxiv_id,
                paper_id=paper_id,
                sections=paper_data.get("sections"),
            )

            if not chunks:
                logger.warning(f"No chunks created for paper {arxiv_id}")
                return {"chunks_created": 0, "chunks_indexed": 0, "embeddings_generated": 0, "errors": 0}

            logger.info(f"Created {len(chunks)} chunks for paper {arxiv_id}")

            # Step 2: Generate embeddings for chunks (supporting Parent-Child chunking)
            from src.config import get_settings
            settings = get_settings()

            chunks_with_embeddings = []
            chunks_created_count = 0

            if getattr(settings, "enable_parent_child", True):
                import uuid
                child_documents = []
                for parent_index, chunk in enumerate(chunks):
                    parent_text = chunk.text
                    parent_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{arxiv_id}_parent_{parent_index}"))
                    
                    text_len = len(parent_text)
                    child_size = 500
                    overlap = 100
                    
                    start = 0
                    child_idx = 0
                    while start < text_len:
                        end = start + child_size
                        child_text = parent_text[start:end]
                        
                        child_documents.append({
                            "parent_id": parent_id,
                            "parent_text": parent_text,
                            "child_text": child_text,
                            "start_char": chunk.metadata.start_char + start,
                            "end_char": chunk.metadata.start_char + min(end, text_len),
                            "chunk_index": parent_index * 100 + child_idx,
                            "section_title": chunk.metadata.section_title,
                            "arxiv_id": chunk.arxiv_id,
                            "paper_id": chunk.paper_id,
                        })
                        
                        if end >= text_len:
                            break
                        start += child_size - overlap
                        child_idx += 1
                        
                if not child_documents:
                    logger.warning(f"No child chunks created for paper {arxiv_id}")
                    return {"chunks_created": 0, "chunks_indexed": 0, "embeddings_generated": 0, "errors": 0}

                # Generate embeddings for child chunks
                child_texts = [doc["child_text"] for doc in child_documents]
                embeddings = await self.embeddings_client.embed_passages(
                    texts=child_texts,
                    batch_size=50,
                )
                
                if len(embeddings) != len(child_documents):
                    logger.error(f"Embedding count mismatch: {len(embeddings)} != {len(child_documents)}")
                    return {"chunks_created": len(child_documents), "chunks_indexed": 0, "embeddings_generated": len(embeddings), "errors": 1}
                
                for doc, embedding in zip(child_documents, embeddings):
                    chunk_data = {
                        "arxiv_id": doc["arxiv_id"],
                        "paper_id": doc["paper_id"],
                        "chunk_index": doc["chunk_index"],
                        "chunk_text": doc["child_text"],
                        "chunk_word_count": len(doc["child_text"].split()),
                        "start_char": doc["start_char"],
                        "end_char": doc["end_char"],
                        "section_title": doc["section_title"],
                        "embedding_model": "jina-embeddings-v3",
                        "parent_id": doc["parent_id"],
                        "parent_text": doc["parent_text"],
                        "title": paper_data.get("title", ""),
                        "authors": ", ".join(paper_data.get("authors", []))
                        if isinstance(paper_data.get("authors"), list)
                        else paper_data.get("authors", ""),
                        "abstract": paper_data.get("abstract", ""),
                        "categories": paper_data.get("categories", []),
                        "published_date": paper_data.get("published_date"),
                    }
                    if tenant_id:
                        chunk_data["tenant_id"] = tenant_id
                        
                    chunks_with_embeddings.append({"chunk_data": chunk_data, "embedding": embedding})
                
                chunks_created_count = len(child_documents)
            else:
                chunk_texts = [chunk.text for chunk in chunks]
                embeddings = await self.embeddings_client.embed_passages(
                    texts=chunk_texts,
                    batch_size=50,
                )
                
                if len(embeddings) != len(chunks):
                    logger.error(f"Embedding count mismatch: {len(embeddings)} != {len(chunks)}")
                    return {"chunks_created": len(chunks), "chunks_indexed": 0, "embeddings_generated": len(embeddings), "errors": 1}
                
                for chunk, embedding in zip(chunks, embeddings):
                    chunk_data = {
                        "arxiv_id": chunk.arxiv_id,
                        "paper_id": chunk.paper_id,
                        "chunk_index": chunk.metadata.chunk_index,
                        "chunk_text": chunk.text,
                        "chunk_word_count": chunk.metadata.word_count,
                        "start_char": chunk.metadata.start_char,
                        "end_char": chunk.metadata.end_char,
                        "section_title": chunk.metadata.section_title,
                        "embedding_model": "jina-embeddings-v3",
                        "title": paper_data.get("title", ""),
                        "authors": ", ".join(paper_data.get("authors", []))
                        if isinstance(paper_data.get("authors"), list)
                        else paper_data.get("authors", ""),
                        "abstract": paper_data.get("abstract", ""),
                        "categories": paper_data.get("categories", []),
                        "published_date": paper_data.get("published_date"),
                    }
                    if tenant_id:
                        chunk_data["tenant_id"] = tenant_id
                        
                    chunks_with_embeddings.append({"chunk_data": chunk_data, "embedding": embedding})
                
                chunks_created_count = len(chunks)

            # Step 4: Index chunks into OpenSearch
            results = self.opensearch_client.bulk_index_chunks(chunks_with_embeddings)

            # Step 5: ColPali Visual Page Indexing
            try:
                from src.services.vision.colpali import ColPaliVisionService
                vision_service = ColPaliVisionService(self.opensearch_client, self.embeddings_client)
                
                import os
                pdf_path = os.path.join(settings.arxiv.pdf_cache_dir, f"{arxiv_id}.pdf")
                
                if os.path.exists(pdf_path):
                    logger.info(f"Processing ColPali visual layout pages for {arxiv_id}...")
                    from docling.document_converter import DocumentConverter
                    converter = DocumentConverter()
                    result = converter.convert(str(pdf_path), max_num_pages=30)
                    doc = result.document
                    
                    page_elements = {}
                    for page_num in range(1, 31):
                        page_elements[page_num] = {
                            "text": "",
                            "text_regions": 0,
                            "tables": 0,
                            "pictures": 0,
                            "equations": 0
                        }
                    
                    # Track elements
                    for text_item in getattr(doc, "texts", []):
                        page_no = getattr(text_item, "prov", None)
                        if page_no and hasattr(page_no, "page_no"):
                            page_no = page_no.page_no
                        else:
                            page_no = getattr(text_item, "page", 1) or 1
                        
                        if isinstance(page_no, int) and page_no in page_elements:
                            page_elements[page_no]["text"] += getattr(text_item, "text", "") + "\n"
                            page_elements[page_no]["text_regions"] += 1
                            
                    for table_item in getattr(doc, "tables", []):
                        page_no = getattr(table_item, "prov", None)
                        if page_no and hasattr(page_no, "page_no"):
                            page_no = page_no.page_no
                        else:
                            page_no = getattr(table_item, "page", 1) or 1
                            
                        if isinstance(page_no, int) and page_no in page_elements:
                            page_elements[page_no]["tables"] += 1
                            
                    for pic_item in getattr(doc, "pictures", []):
                        page_no = getattr(pic_item, "prov", None)
                        if page_no and hasattr(page_no, "page_no"):
                            page_no = page_no.page_no
                        else:
                            page_no = getattr(pic_item, "page", 1) or 1
                            
                        if isinstance(page_no, int) and page_no in page_elements:
                            page_elements[page_no]["pictures"] += 1
                            
                    max_page_found = 1
                    for page_num in page_elements:
                        stats = page_elements[page_num]
                        if stats["text"].strip() or stats["tables"] > 0 or stats["pictures"] > 0:
                            max_page_found = max(max_page_found, page_num)
                            
                    for page_num in range(1, max_page_found + 1):
                        stats = page_elements[page_num]
                        await vision_service.index_visual_page(
                            arxiv_id=arxiv_id,
                            paper_id=paper_id,
                            page_number=page_num,
                            page_text=stats["text"],
                            layout_stats={
                                "text_regions": stats["text_regions"],
                                "tables": stats["tables"],
                                "pictures": stats["pictures"],
                                "equations": stats["equations"]
                            },
                            tenant_id=tenant_id or "default"
                        )
                else:
                    logger.info(f"PDF not found at {pdf_path}. Indexing abstract as fallback visual page 1 for {arxiv_id}")
                    await vision_service.index_visual_page(
                        arxiv_id=arxiv_id,
                        paper_id=paper_id,
                        page_number=1,
                        page_text=paper_data.get("abstract", "") or paper_data.get("title", ""),
                        layout_stats={"text_regions": 5, "tables": 0, "pictures": 0, "equations": 0},
                        tenant_id=tenant_id or "default"
                    )
            except Exception as e:
                logger.error(f"Failed to process visual pages for paper {arxiv_id}: {e}")

            # Optional multimodal indexing of figures/images if enabled
            from src.config import get_settings
            settings = get_settings()
            if getattr(settings, "enable_multimodal", False):
                import os
                pdf_path = os.path.join(settings.arxiv.pdf_cache_dir, f"{arxiv_id}.pdf")
                if os.path.exists(pdf_path):
                    logger.info(f"Multimodal indexing enabled. Processing figures for paper {arxiv_id}...")
                    try:
                        from src.services.multimodal import MultiModalProcessor
                        from src.services.ollama.factory import make_ollama_client
                        processor = MultiModalProcessor()
                        ollama_client = make_ollama_client()
                        
                        figures_indexed = await processor.process_and_index_pdf_figures(
                            pdf_path=pdf_path,
                            paper_id=arxiv_id,
                            opensearch_client=self.opensearch_client,
                            ollama_client=ollama_client,
                            embeddings_client=self.embeddings_client,
                            tenant_id=tenant_id,
                        )
                        logger.info(f"Successfully processed & indexed {figures_indexed} figures/images for paper {arxiv_id}")
                    except Exception as e:
                        logger.error(f"Multimodal figures processing failed for paper {arxiv_id}: {e}")
                else:
                    logger.debug(f"No PDF found at {pdf_path} for multimodal figure indexing of {arxiv_id}")

            logger.info(f"Indexed paper {arxiv_id}: {results['success']} chunks successful, {results['failed']} failed")

            return {
                "chunks_created": chunks_created_count,
                "chunks_indexed": results["success"],
                "embeddings_generated": len(embeddings),
                "errors": results["failed"],
            }

        except Exception as e:
            logger.error(f"Error indexing paper {arxiv_id}: {e}")
            return {"chunks_created": 0, "chunks_indexed": 0, "embeddings_generated": 0, "errors": 1}

    async def index_papers_batch(self, papers: List[Dict], replace_existing: bool = False) -> Dict[str, int]:
        """Index multiple papers in batch.

        :param papers: List of paper data
        :param replace_existing: If True, delete existing chunks before indexing
        :returns: Aggregated statistics
        """
        total_stats = {
            "papers_processed": 0,
            "total_chunks_created": 0,
            "total_chunks_indexed": 0,
            "total_embeddings_generated": 0,
            "total_errors": 0,
        }

        for paper in papers:
            arxiv_id = paper.get("arxiv_id")

            # Optionally delete existing chunks
            if replace_existing and arxiv_id:
                self.opensearch_client.delete_paper_chunks(arxiv_id)

            # Index the paper
            stats = await self.index_paper(paper)

            # Update totals
            total_stats["papers_processed"] += 1
            total_stats["total_chunks_created"] += stats["chunks_created"]
            total_stats["total_chunks_indexed"] += stats["chunks_indexed"]
            total_stats["total_embeddings_generated"] += stats["embeddings_generated"]
            total_stats["total_errors"] += stats["errors"]

        logger.info(
            f"Batch indexing complete: {total_stats['papers_processed']} papers, "
            f"{total_stats['total_chunks_indexed']} chunks indexed"
        )

        return total_stats

    async def reindex_paper(self, arxiv_id: str, paper_data: Dict) -> Dict[str, int]:
        """Reindex a paper by deleting old chunks and creating new ones.

        :param arxiv_id: ArXiv ID of the paper
        :param paper_data: Updated paper data
        :returns: Indexing statistics
        """
        # Delete existing chunks
        deleted = self.opensearch_client.delete_paper_chunks(arxiv_id)
        if deleted:
            logger.info(f"Deleted existing chunks for paper {arxiv_id}")

        # Index with new data
        return await self.index_paper(paper_data)
