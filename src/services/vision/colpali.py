import logging
import io
import os
import time
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image, ImageDraw

from src.services.opensearch.index_config_hybrid import ARXIV_PAPER_VISUAL_PAGES_INDEX

logger = logging.getLogger(__name__)

class ColPaliVisionService:
    """Simulated ColPali visual page indexing and dense vector retrieval service.

    Builds page layout screenshots using Pillow and generates 512-dimensional
    normalized embeddings representing page text, structures, and layout density.
    """

    def __init__(self, opensearch_client, embeddings_client):
        """Initialize ColPali service.

        :param opensearch_client: Unified OpenSearch client
        :param embeddings_client: Embeddings client (e.g. Jina embeddings)
        """
        self.opensearch = opensearch_client
        self.embeddings = embeddings_client
        
        # Ensure static output directories exist
        self.pages_dir = Path("static/data/pages")
        self.pages_dir.mkdir(parents=True, exist_ok=True)

    def draw_page_layout(
        self,
        arxiv_id: str,
        page_num: int,
        paragraphs: int,
        tables: int,
        pictures: int,
        equations: int,
    ) -> str:
        """Draws a visual representation of the page layout using Pillow.

        Saves the layout image to static directory and returns the relative URL path.
        """
        width, height = 600, 800
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)

        # Draw margins
        draw.rectangle([10, 10, width - 10, height - 10], outline="#ddd", width=1)

        # Draw running header / page number
        draw.text((20, 20), f"arXiv ID: {arxiv_id}", fill="#888")
        draw.text((width - 80, 20), f"Page {page_num}", fill="#888")

        # Draw a simulated title block on the first page
        y_cursor = 60
        if page_num == 1:
            draw.rectangle([40, y_cursor, width - 40, y_cursor + 40], fill="#eee", outline="#ccc")
            # draw abstract block
            draw.rectangle([60, y_cursor + 60, width - 60, y_cursor + 140], fill="#f9f9f9", outline="#ddd")
            y_cursor += 160

        # Draw pictures (Figures) in green/orange
        for i in range(min(pictures, 3)):
            box = [50 + i * 40, y_cursor, width - 50 - i * 40, y_cursor + 120]
            draw.rectangle(box, fill="#e8f5e9", outline="#4caf50", width=2)
            draw.text((box[0] + 10, box[1] + 10), f"Figure {i+1} [Visual Layout]", fill="#2e7d32")
            y_cursor += 140

        # Draw tables in blue
        for i in range(min(tables, 2)):
            box = [60, y_cursor, width - 60, y_cursor + 100]
            draw.rectangle(box, fill="#e3f2fd", outline="#2196f3", width=2)
            # Draw table grids
            for col in range(1, 5):
                x = box[0] + col * (box[2] - box[0]) // 5
                draw.line([x, box[1], x, box[3]], fill="#bbdefb")
            for row in range(1, 4):
                y = box[1] + row * (box[3] - box[1]) // 4
                draw.line([box[0], y, box[2], y], fill="#bbdefb")
            draw.text((box[0] + 10, box[1] + 10), f"Table {i+1} [Visual Data]", fill="#1565c0")
            y_cursor += 120

        # Draw equations in thin red lines
        for i in range(min(equations, 4)):
            box = [150, y_cursor, width - 150, y_cursor + 25]
            draw.rectangle(box, fill="#ffebee", outline="#f44336", width=1)
            draw.text((box[0] + 10, box[1] + 5), f"Equation {i+1} (Math block)", fill="#c62828")
            y_cursor += 40

        # Draw standard text lines (gray rectangles)
        for i in range(min(paragraphs, 8)):
            if y_cursor > height - 60:
                break
            # Draw double column layout
            left_col = [40, y_cursor, width // 2 - 20, y_cursor + 12]
            right_col = [width // 2 + 20, y_cursor, width - 40, y_cursor + 12]
            draw.rectangle(left_col, fill="#f5f5f5")
            draw.rectangle(right_col, fill="#f5f5f5")
            y_cursor += 20

        # Save image to pages folder
        filename = f"{arxiv_id}_page_{page_num}.png"
        filepath = self.pages_dir / filename
        image.save(filepath, format="PNG")
        
        return f"/static/data/pages/{filename}"

    async def generate_visual_embedding(self, page_text: str, layout_stats: Dict[str, int]) -> List[float]:
        """Generates a 512-dimension normalized visual-semantic embedding.

        Projects the text embeddings + layout statistics using a deterministic projection.
        """
        # Get 1024-dimensional query/document embedding first
        raw_emb = await self.embeddings.embed_query(page_text or "empty page")
        if raw_emb is None:
            # Fallback mock embedding when Jina API is missing
            raw_emb = [1.0] + [0.0] * 1023
            
        raw_vector = np.array(raw_emb, dtype=np.float32)

        # Create deterministic pseudo-random projection matrix from 1024 -> 512
        # Seed by first few dimensions of the embedding to remain consistent per run
        np.random.seed(int(abs(raw_vector[0] * 100000)) % 65535)
        projection_matrix = np.random.randn(1024, 512).astype(np.float32)

        # Project vector to 512
        projected = np.dot(raw_vector, projection_matrix)

        # Add visual bias based on layout elements
        # (Equations, figures, tables are added to specific feature buckets to bias the matching)
        projected[0] += float(layout_stats.get("tables", 0)) * 0.5
        projected[1] += float(layout_stats.get("pictures", 0)) * 0.5
        projected[2] += float(layout_stats.get("equations", 0)) * 0.3

        # L2 Normalize the final 512 vector
        norm = np.linalg.norm(projected)
        if norm > 0:
            projected = projected / norm

        return projected.tolist()

    async def index_visual_page(
        self,
        arxiv_id: str,
        paper_id: str,
        page_number: int,
        page_text: str,
        layout_stats: Dict[str, int],
        tenant_id: str = "default",
    ) -> bool:
        """Generates the layout image, visual embedding, and indexes the page into OpenSearch."""
        try:
            image_path = self.draw_page_layout(
                arxiv_id=arxiv_id,
                page_num=page_number,
                paragraphs=layout_stats.get("text_regions", 5),
                tables=layout_stats.get("tables", 0),
                pictures=layout_stats.get("pictures", 0),
                equations=layout_stats.get("equations", 0),
            )

            visual_embedding = await self.generate_visual_embedding(page_text, layout_stats)

            doc_body = {
                "page_id": f"{arxiv_id}_p{page_number}",
                "arxiv_id": arxiv_id,
                "paper_id": paper_id,
                "tenant_id": tenant_id,
                "page_number": page_number,
                "image_path": image_path,
                "page_text": page_text,
                "visual_embedding": visual_embedding,
                "layout_stats": layout_stats,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }

            self.opensearch.client.index(
                index=ARXIV_PAPER_VISUAL_PAGES_INDEX,
                body=doc_body,
                id=f"{arxiv_id}_p{page_number}",
                refresh=True,
            )
            logger.info(f"Indexed visual page {page_number} for paper {arxiv_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to index visual page {page_number} for paper {arxiv_id}: {e}")
            return False

    async def search_visual_pages(
        self,
        query: str,
        top_k: int = 4,
        tenant_id: str = "default",
    ) -> List[Dict[str, Any]]:
        """Queries the visual page index using dense cosine similarity."""
        try:
            # Generate search embedding
            # (We set dummy layout statistics for the search query)
            query_emb = await self.generate_visual_embedding(query, {"tables": 0, "pictures": 0, "equations": 0})

            # Check if visual page index exists
            if not self.opensearch.client.indices.exists(index=ARXIV_PAPER_VISUAL_PAGES_INDEX):
                logger.warning(f"Index {ARXIV_PAPER_VISUAL_PAGES_INDEX} does not exist yet")
                return []

            search_query = {
                "size": top_k,
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"tenant_id": tenant_id}}
                        ],
                        "should": [
                            # BM25 match to score layout text
                            {"match": {"page_text": {"query": query, "boost": 1.5}}},
                            # kNN dense vector visual layout match
                            {
                                "knn": {
                                    "visual_embedding": {
                                        "vector": query_emb,
                                        "k": top_k,
                                        "boost": 2.5
                                    }
                                }
                            }
                        ]
                    }
                }
            }

            response = self.opensearch.client.search(
                index=ARXIV_PAPER_VISUAL_PAGES_INDEX,
                body=search_query
            )

            hits = []
            for hit in response.get("hits", {}).get("hits", []):
                source = hit["_source"]
                hits.append({
                    "arxiv_id": source.get("arxiv_id"),
                    "page_number": source.get("page_number"),
                    "image_path": source.get("image_path"),
                    "score": hit["_score"],
                    "page_text": source.get("page_text")[:300] + "...",
                    "layout_stats": source.get("layout_stats"),
                })

            return hits

        except Exception as e:
            logger.error(f"Error querying ColPali visual search: {e}")
            return []
