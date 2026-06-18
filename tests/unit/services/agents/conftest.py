import pytest
from unittest.mock import Mock, AsyncMock
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from src.services.agents.context import Context
from src.services.agents.models import GuardrailScoring

@pytest.fixture
def mock_opensearch_client():
    client = Mock()
    client.search_unified = Mock(return_value={
        "hits": [
            {
                "chunk_text": "Transformers are neural network architectures based on self-attention mechanisms.",
                "arxiv_id": "1706.03762",
                "title": "Attention Is All You Need",
                "authors": "Vaswani et al.",
                "score": 0.95,
                "section_name": "Abstract"
            },
            {
                "chunk_text": "Second document chunk content.",
                "arxiv_id": "1810.04805",
                "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                "authors": "Devlin et al.",
                "score": 0.90,
                "section_name": "Introduction"
            }
        ]
    })
    return client

@pytest.fixture
def mock_ollama_client():
    client = Mock()
    client.generate_rag_answer = AsyncMock(return_value={"answer": "Mocked answer"})
    client.generate_rag_answer_stream = AsyncMock()
    return client

@pytest.fixture
def mock_jina_embeddings_client():
    client = Mock()
    client.embed_query = AsyncMock(return_value=[0.1] * 1024)
    return client

@pytest.fixture
def test_context(mock_ollama_client, mock_opensearch_client, mock_jina_embeddings_client):
    context = Context(
        ollama_client=mock_ollama_client,
        opensearch_client=mock_opensearch_client,
        embeddings_client=mock_jina_embeddings_client,
        langfuse_tracer=None,
        trace=None,
        langfuse_enabled=False,
        model_name="llama3.2:1b",
        temperature=0.0,
        top_k=3,
        max_retrieval_attempts=2,
        guardrail_threshold=60
    )
    return context

@pytest.fixture
def sample_human_message():
    return HumanMessage(content="What is machine learning?")

@pytest.fixture
def sample_tool_message():
    return ToolMessage(
        content="Transformers are neural network architectures based on self-attention mechanisms.",
        tool_call_id="call_1"
    )

@pytest.fixture
def sample_ai_message():
    return AIMessage(content="Machine learning is a subset of AI.")
