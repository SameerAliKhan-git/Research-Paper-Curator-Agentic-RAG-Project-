import base64
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MultiModalProcessor:
    """Extracts images from PDFs, describes them with a vision LLM, and indexes the captions."""

    def extract_images_from_pdf(self, pdf_path: str | Path) -> List[Dict[str, Any]]:
        """Extract images from a PDF using docling.

        Returns a list of dicts with keys: page, image_bytes, image_index.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            logger.error(f"PDF not found: {pdf_path}")
            return []

        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption

            pipeline_options = PdfPipelineOptions(do_table_structure=False)
            converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})
            result = converter.convert(str(pdf_path), max_num_pages=30)
            doc = result.document

            images: List[Dict[str, Any]] = []
            image_index = 0

            for element in doc.pictures:
                if hasattr(element, "image") and element.image is not None:
                    img = element.image
                    if hasattr(img, "pil_image") and img.pil_image is not None:
                        import io

                        buf = io.BytesIO()
                        img.pil_image.save(buf, format="PNG")
                        images.append(
                            {
                                "page": getattr(element, "page", 0),
                                "image_bytes": buf.getvalue(),
                                "image_index": image_index,
                            }
                        )
                        image_index += 1

            logger.info(f"Extracted {len(images)} images from {pdf_path.name}")
            return images

        except ImportError:
            logger.warning("docling not installed; image extraction unavailable")
            return []
        except Exception as e:
            logger.error(f"Image extraction failed for {pdf_path}: {e}")
            return []

    async def describe_image_with_llm(
        self,
        image_path: Optional[str | Path] = None,
        ollama_client: Any = None,
        model: str = "llava:latest",
        image_bytes: Optional[bytes] = None,
    ) -> str:
        """Use an Ollama vision model to describe an image.

        Args:
            image_path: Path to the image file (optional if image_bytes provided).
            ollama_client: OllamaClient instance.
            model: Vision model name.
            image_bytes: Raw bytes of the image (optional if image_path provided).

        Returns:
            Text description of the image.
        """
        if not image_bytes:
            if not image_path:
                logger.error("Either image_path or image_bytes must be provided")
                return ""
            path = Path(image_path)
            if not path.exists():
                logger.error(f"Image not found: {path}")
                return ""
            try:
                image_bytes = path.read_bytes()
            except Exception as e:
                logger.error(f"Failed to read image from path {path}: {e}")
                return ""

        try:
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            client = await ollama_client.get_client()

            data = {
                "model": model,
                "prompt": "Describe this figure from a research paper in detail. "
                "Include the type of chart/plot, axes, key data trends, "
                "and any notable observations.",
                "images": [image_b64],
                "stream": False,
            }

            response = await client.post(f"{ollama_client.base_url}/api/generate", json=data)

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")

            logger.error(f"Vision model returned status {response.status_code}")
            return ""

        except Exception as e:
            logger.error(f"Image description failed: {e}")
            return ""

    def index_image_caption(
        self,
        caption: str,
        paper_id: str,
        opensearch_client: Any,
        image_index: int = 0,
        embedding: Optional[List[float]] = None,
        tenant_id: Optional[str] = None,
    ) -> bool:
        """Index an image caption as a chunk in OpenSearch.

        Uses the image_index field to distinguish from text chunks.
        """
        chunk_data: Dict[str, Any] = {
            "arxiv_id": paper_id,
            "paper_id": paper_id,
            "chunk_text": caption,
            "chunk_index": image_index,
            "chunk_word_count": len(caption.split()),
            "image_index": image_index,
            "section_title": "figure",
        }

        if tenant_id:
            chunk_data["tenant_id"] = tenant_id

        if embedding:
            chunk_data["embedding"] = embedding
            try:
                opensearch_client.client.index(
                    index=opensearch_client.index_name,
                    body=chunk_data,
                    refresh=False,
                )
                return True
            except Exception as e:
                logger.error(f"Failed to index image caption with embedding: {e}")
                return False

        try:
            opensearch_client.client.index(
                index=opensearch_client.index_name,
                body=chunk_data,
                refresh=False,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to index image caption: {e}")
            return False

    async def process_and_index_pdf_figures(
        self,
        pdf_path: str | Path,
        paper_id: str,
        opensearch_client: Any,
        ollama_client: Any,
        embeddings_client: Any = None,
        model: str = "llava:latest",
        tenant_id: Optional[str] = None,
    ) -> int:
        """Extract all images from a PDF, describe them using Ollama, and index the captions in OpenSearch.

        Returns the number of successfully indexed figure captions.
        """
        images = self.extract_images_from_pdf(pdf_path)
        if not images:
            return 0

        indexed_count = 0
        for img in images:
            page = img["page"]
            img_bytes = img["image_bytes"]
            idx = img["image_index"]

            logger.info(f"Generating caption for figure {idx} on page {page} of paper {paper_id}...")
            caption = await self.describe_image_with_llm(
                ollama_client=ollama_client,
                model=model,
                image_bytes=img_bytes,
            )

            if not caption:
                logger.warning(f"Failed to generate description for figure {idx} on page {page}")
                continue

            full_caption = f"Figure on Page {page} of Paper: {caption}"

            # Optional: generate embedding for the caption
            embedding = None
            if embeddings_client:
                try:
                    embeddings = await embeddings_client.embed_passages([full_caption])
                    if embeddings:
                        embedding = embeddings[0]
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for figure {idx} caption: {e}")

            success = self.index_image_caption(
                caption=full_caption,
                paper_id=paper_id,
                opensearch_client=opensearch_client,
                image_index=idx,
                embedding=embedding,
                tenant_id=tenant_id,
            )
            if success:
                indexed_count += 1

        logger.info(f"Successfully processed and indexed {indexed_count}/{len(images)} figures for paper {paper_id}")
        return indexed_count
