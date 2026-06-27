import pytest
from src.services.agents.agentic_rag import AgenticRAGService
from src.services.agents.config import GraphConfig
from src.services.opensearch.factory import make_opensearch_client
from src.services.embeddings.factory import make_embeddings_service
from src.services.ollama.factory import make_ollama_client


def is_services_reachable() -> bool:
    try:
        opensearch = make_opensearch_client()
        if not opensearch.health_check():
            return False
        return True
    except Exception:
        return False


@pytest.mark.skipif(not is_services_reachable(), reason="Database/OpenSearch services not running")
@pytest.mark.asyncio
async def test_agentic_rag_service_ask():
    opensearch = make_opensearch_client()
    embeddings = make_embeddings_service()
    ollama = make_ollama_client()

    service = AgenticRAGService(
        opensearch_client=opensearch,
        ollama_client=ollama,
        embeddings_client=embeddings,
        graph_config=GraphConfig(model="llama3.2:1b")
    )

    # Test executing a simple query
    result = await service.ask("What are transformer architectures?", user_id="test_user")

    assert "answer" in result
    assert "sources" in result
    assert isinstance(result["sources"], list)
    assert result.get("search_mode") == "hybrid"
