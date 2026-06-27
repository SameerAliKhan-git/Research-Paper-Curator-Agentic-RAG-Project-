from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient
from src.main import app


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    """Async backend for testing."""
    return "asyncio"


@pytest.fixture
async def client():
    """HTTP client for API testing with mocked services."""
    with (
        patch("src.database.make_database") as mock_make_db,
        patch("src.main.make_opensearch_client") as mock_make_os,
        patch("src.main.make_arxiv_client") as mock_make_arxiv,
        patch("src.main.make_pdf_parser_service") as mock_make_pdf,
        patch("src.main.make_embeddings_service") as mock_make_embeddings,
        patch("src.main.make_ollama_client") as mock_make_ollama,
        patch("src.main.make_langfuse_tracer") as mock_make_langfuse,
        patch("src.main.make_cache_client") as mock_make_cache,
        patch("src.main.make_telegram_service") as mock_make_telegram,
        patch("src.main.make_semantic_cache") as mock_make_semantic,
        patch("src.main.make_api_key_service") as mock_make_apikey,
        patch("src.main.make_reranker_service") as mock_make_reranker,
        patch("src.repositories.paper.PaperRepository.get_by_arxiv_id") as mock_get_by_id,
    ):
        # Mock database
        mock_db = MagicMock()
        mock_db.get_session.return_value.__enter__.return_value = MagicMock()
        mock_db.get_session.return_value.__exit__.return_value = None
        mock_make_db.return_value = mock_db

        # Mock OpenSearch
        mock_os = MagicMock()
        mock_os.health_check.return_value = True
        mock_os.setup_indices.return_value = {"hybrid_index": True}
        mock_os.client.count.return_value = {"count": 10}
        
        # Unified search mock returning sample hit
        mock_os.search_unified = AsyncMock(
            return_value={
                "total": 1,
                "hits": [
                    {
                        "arxiv_id": "2301.00001",
                        "title": "Test Paper",
                        "chunk_text": "This is a test paper.",
                        "score": 1.0,
                        "abstract": "This is a test paper abstract.",
                    }
                ],
            }
        )
        mock_os.get_index_stats.return_value = {
            "index_name": "arxiv-papers",
            "document_count": 10,
        }
        mock_make_os.return_value = mock_os

        # Mock ArXiv
        mock_arxiv = MagicMock()
        mock_make_arxiv.return_value = mock_arxiv

        # Mock PDF Parser
        mock_pdf = MagicMock()
        mock_make_pdf.return_value = mock_pdf

        # Mock Embeddings
        mock_embeddings = AsyncMock()
        mock_embeddings.embed_query.return_value = [0.1] * 1024
        mock_make_embeddings.return_value = mock_embeddings

        # Mock Ollama
        mock_ollama = AsyncMock()
        mock_ollama.health_check.return_value = {"status": "healthy", "version": "0.3.0", "message": "Ollama service is running"}
        mock_ollama.generate_rag_answer.return_value = {"answer": "Mocked answer", "done": True}
        
        async def mock_generate_stream(*args, **kwargs):
            yield {"response": "Mocked", "done": False}
            yield {"response": " answer", "done": True}
        mock_ollama.generate_rag_answer_stream = mock_generate_stream
        mock_make_ollama.return_value = mock_ollama

        # Mock Langfuse
        mock_langfuse = MagicMock()
        mock_make_langfuse.return_value = mock_langfuse

        # Mock Cache
        mock_make_cache.return_value = None

        # Mock Telegram
        mock_make_telegram.return_value = None

        # Mock Semantic Cache
        mock_make_semantic.return_value = None

        # Mock API Key Service
        mock_make_apikey.return_value = None

        # Mock Reranker
        mock_make_reranker.return_value = None

        # Mock repository
        mock_get_by_id.return_value = None

        from src.services.auth.api_key_service import require_api_key, APIKeyMetadata
        app.dependency_overrides[require_api_key] = lambda: APIKeyMetadata(
            key_hash="test_hash",
            user_id="test_user",
            tier="admin",
            rate_limit=1000,
            quota_remaining=1000,
            tenants=["default"],
        )
        try:
            async with LifespanManager(app) as manager:
                async with AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as client:
                    yield client
        finally:
            app.dependency_overrides.clear()
