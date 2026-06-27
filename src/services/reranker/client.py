"""Reranker client implementations for Jina AI, Cohere, and BGE."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel
from src.config import RerankerSettings, get_settings
from src.services.circuit_breaker import async_circuit_breaker_retry

logger = logging.getLogger(__name__)


class RerankResult(BaseModel):
    """Result of a reranking operation."""

    index: int
    relevance_score: float
    document: Dict[str, Any]


class RerankerClient(ABC):
    """Abstract base class for reranker clients."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_n: Optional[int] = None,
    ) -> List[RerankResult]:
        """Rerank documents by relevance to query."""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Check if reranker service is available."""
        pass

    async def close(self):
        """Cleanup resources. Override in subclasses."""
        pass


class JinaRerankerClient(RerankerClient):
    """Jina AI Reranker client."""

    def __init__(self, settings: RerankerSettings):
        self.settings = settings
        self.api_key = settings.api_key
        self.base_url = settings.base_url
        self.model = settings.model
        self.timeout = settings.timeout

        self._client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.timeout,
        )

    @async_circuit_breaker_retry(
        service_name="jina_reranker",
        max_retries=3,
        retry_exceptions=(httpx.HTTPStatusError, httpx.HTTPError),
        failure_threshold=5,
        recovery_timeout=30,
    )
    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_n: Optional[int] = None,
    ) -> List[RerankResult]:
        """Rerank using Jina AI Reranker API."""
        if not documents:
            return []

        top_n = top_n or self.settings.top_n
        top_n = min(top_n, len(documents))

        # Extract text from documents
        texts = []
        for doc in documents:
            text = doc.get("chunk_text", doc.get("content", doc.get("text", "")))
            texts.append(text)

        payload = {
            "model": self.model,
            "query": query,
            "documents": texts,
            "top_n": top_n,
            "return_documents": True,
        }

        response = await self._client.post(f"{self.base_url}", json=payload)
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("results", []):
            idx = item["index"]
            results.append(
                RerankResult(
                    index=idx,
                    relevance_score=item["relevance_score"],
                    document=documents[idx],
                )
            )

        logger.debug(f"Jina reranked {len(documents)} docs -> top {len(results)}")
        return results

    def health_check(self) -> bool:
        """Check Jina API availability."""
        try:
            response = httpx.get(
                "https://api.jina.ai/v1/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=5.0,
            )
            return response.status_code == 200
        except Exception:
            return False

    async def close(self):
        await self._client.aclose()


class CohereRerankerClient(RerankerClient):
    """Cohere Reranker client."""

    def __init__(self, settings: RerankerSettings):
        self.settings = settings
        self.api_key = settings.api_key
        self.model = settings.model
        self.base_url = "https://api.cohere.ai/v1/rerank"
        self.timeout = settings.timeout

        self._client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.timeout,
        )

    @async_circuit_breaker_retry(
        service_name="cohere_reranker",
        max_retries=3,
        retry_exceptions=(httpx.HTTPStatusError, httpx.HTTPError),
        failure_threshold=5,
        recovery_timeout=30,
    )
    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_n: Optional[int] = None,
    ) -> List[RerankResult]:
        """Rerank using Cohere Reranker API."""
        if not documents:
            return []

        top_n = top_n or self.settings.top_n
        top_n = min(top_n, len(documents))

        texts = []
        for doc in documents:
            text = doc.get("chunk_text", doc.get("content", doc.get("text", "")))
            texts.append(text)

        payload = {
            "model": self.model,
            "query": query,
            "documents": texts,
            "top_n": top_n,
            "return_documents": True,
        }

        response = await self._client.post(self.base_url, json=payload)
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("results", []):
            idx = item["index"]
            results.append(
                RerankResult(
                    index=idx,
                    relevance_score=item["relevance_score"],
                    document=documents[idx],
                )
            )

        logger.debug(f"Cohere reranked {len(documents)} docs -> top {len(results)}")
        return results

    def health_check(self) -> bool:
        """Check Cohere API availability."""
        try:
            response = httpx.get(
                "https://api.cohere.ai/v1/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=5.0,
            )
            return response.status_code == 200
        except Exception:
            return False

    async def close(self):
        await self._client.aclose()


class BGERerankerClient(RerankerClient):
    """Local BGE Reranker client (via sentence-transformers)."""

    def __init__(self, settings: RerankerSettings):
        self.settings = settings
        self.model_name = settings.model or "BAAI/bge-reranker-v2-m3"
        self._model = None
        self._device = "cpu"

    def _load_model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                import torch
                from sentence_transformers import CrossEncoder

                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                self._model = CrossEncoder(self.model_name, device=self._device)
                logger.info(f"Loaded BGE reranker: {self.model_name} on {self._device}")
            except ImportError:
                logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
                raise
            except Exception as e:
                logger.error(f"Failed to load BGE reranker: {e}")
                raise

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_n: Optional[int] = None,
    ) -> List[RerankResult]:
        """Rerank using local BGE CrossEncoder with timeout protection."""
        if not documents:
            return []

        self._load_model()
        top_n = top_n or self.settings.top_n
        top_n = min(top_n, len(documents))

        texts = []
        for doc in documents:
            text = doc.get("chunk_text", doc.get("content", doc.get("text", "")))
            texts.append(text)

        try:
            import asyncio

            # Create query-document pairs
            pairs = [(query, text) for text in texts]

            # Run prediction with timeout protection
            async def _predict_with_timeout():
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None,
                    lambda: self._model.predict(pairs, show_progress_bar=False),
                )

            scores = await asyncio.wait_for(_predict_with_timeout(), timeout=30.0)

            # Sort by score descending
            ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

            results = []
            for idx, score in ranked[:top_n]:
                results.append(
                    RerankResult(
                        index=idx,
                        relevance_score=float(score),
                        document=documents[idx],
                    )
                )

            logger.debug(f"BGE reranked {len(documents)} docs -> top {len(results)}")
            return results

        except asyncio.TimeoutError:
            logger.warning("BGE reranker timed out after 30s, using original order")
            return []
        except Exception as e:
            logger.warning(f"BGE reranker error: {e}, using original order")
            return []

    def health_check(self) -> bool:
        """Check if model can be loaded."""
        try:
            self._load_model()
            return self._model is not None
        except Exception:
            return False

    async def close(self):
        """Cleanup."""
        if self._model is not None:
            del self._model
            self._model = None


def create_reranker_client(settings: RerankerSettings) -> RerankerClient:
    """Factory function to create appropriate reranker client."""
    provider = settings.provider.lower()

    if provider == "jina":
        if not settings.api_key:
            raise ValueError("Jina API key required for Jina reranker")
        return JinaRerankerClient(settings)
    elif provider == "cohere":
        if not settings.api_key:
            raise ValueError("Cohere API key required for Cohere reranker")
        return CohereRerankerClient(settings)
    elif provider == "bge":
        return BGERerankerClient(settings)
    else:
        raise ValueError(f"Unknown reranker provider: {provider}")
