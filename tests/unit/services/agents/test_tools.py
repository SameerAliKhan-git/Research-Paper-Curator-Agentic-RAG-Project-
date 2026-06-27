import pytest
from unittest.mock import AsyncMock
from langchain_core.documents import Document

from src.services.agents.tools import create_retriever_tool, create_arxiv_search_tool, google_search


@pytest.mark.asyncio
async def test_create_retriever_tool_basic(mock_opensearch_client, mock_jina_embeddings_client):
    """Test basic retriever tool creation and invocation."""
    tool = create_retriever_tool(
        opensearch_client=mock_opensearch_client,
        embeddings_client=mock_jina_embeddings_client,
        top_k=2,
        use_hybrid=True,
    )

    # Verify tool properties
    assert tool.name == "retrieve_papers"
    assert "Search and return relevant arXiv research papers" in tool.description

    # Invoke tool
    result = await tool.ainvoke({"query": "machine learning"})

    # Verify result
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(doc, Document) for doc in result)

    # Verify first document
    first_doc = result[0]
    assert first_doc.page_content == "Transformers are neural network architectures based on self-attention mechanisms."
    assert first_doc.metadata["arxiv_id"] == "1706.03762"
    assert first_doc.metadata["title"] == "Attention Is All You Need"
    assert first_doc.metadata["score"] == 0.95

    # Verify embeddings were generated
    mock_jina_embeddings_client.embed_query.assert_called_once_with("machine learning")

    # Verify search was called correctly
    mock_opensearch_client.search_unified.assert_called_once()
    call_args = mock_opensearch_client.search_unified.call_args
    assert call_args.kwargs["query"] == "machine learning"
    assert call_args.kwargs["size"] == 2  # search_unified uses 'size', not 'top_k'
    assert call_args.kwargs["use_hybrid"] is True


@pytest.mark.asyncio
async def test_retriever_tool_empty_results(mock_opensearch_client, mock_jina_embeddings_client):
    """Test retriever tool with no results."""
    from unittest.mock import AsyncMock
    mock_opensearch_client.search_unified = AsyncMock(return_value={"hits": []})

    tool = create_retriever_tool(
        opensearch_client=mock_opensearch_client,
        embeddings_client=mock_jina_embeddings_client,
    )

    result = await tool.ainvoke({"query": "nonexistent topic"})

    assert isinstance(result, list)
    assert len(result) == 0


@pytest.mark.asyncio
async def test_retriever_tool_custom_top_k(mock_opensearch_client, mock_jina_embeddings_client):
    """Test retriever tool with custom top_k parameter."""
    tool = create_retriever_tool(
        opensearch_client=mock_opensearch_client,
        embeddings_client=mock_jina_embeddings_client,
        top_k=5,
        use_hybrid=False,
    )

    await tool.ainvoke({"query": "test query"})

    call_args = mock_opensearch_client.search_unified.call_args
    # search_unified uses 'size' parameter, not 'top_k'
    assert call_args.kwargs["size"] == 5
    assert call_args.kwargs["use_hybrid"] is False


@pytest.mark.asyncio
async def test_retriever_tool_metadata_fields(mock_opensearch_client, mock_jina_embeddings_client):
    """Test that all expected metadata fields are present."""
    from unittest.mock import AsyncMock
    mock_opensearch_client.search_unified = AsyncMock(return_value={
        "hits": [
            {
                "chunk_text": "Test content",
                "arxiv_id": "2301.00001",
                "title": "Test Paper",
                "authors": "Author One, Author Two",
                "score": 0.95,
                "section_name": "Introduction",
            }
        ]
    })

    tool = create_retriever_tool(
        opensearch_client=mock_opensearch_client,
        embeddings_client=mock_jina_embeddings_client,
    )

    result = await tool.ainvoke({"query": "test"})

    doc = result[0]
    assert "arxiv_id" in doc.metadata
    assert "title" in doc.metadata
    assert "authors" in doc.metadata
    assert "score" in doc.metadata
    assert "source" in doc.metadata
    assert "section" in doc.metadata


@pytest.mark.asyncio
async def test_create_arxiv_search_tool(mock_arxiv_client):
    """Test live arXiv search tool creation and invocation."""
    from src.schemas.arxiv.paper import ArxivPaper
    
    mock_paper = ArxivPaper(
        arxiv_id="2301.00002",
        title="Live Paper Title",
        authors=["Author Alice", "Author Bob"],
        abstract="Live abstract content about neural network designs.",
        published_date="2023-01-01",
        categories=["cs.AI"],
        pdf_url="https://arxiv.org/pdf/2301.00002.pdf"
    )
    mock_arxiv_client.fetch_papers_with_query = AsyncMock(return_value=[mock_paper])
    
    tool = create_arxiv_search_tool(
        arxiv_client=mock_arxiv_client,
        top_k=1,
    )
    
    assert tool.name == "search_arxiv"
    
    result = await tool.ainvoke({"query": "deep learning"})
    
    assert isinstance(result, list)
    assert len(result) == 1
    doc = result[0]
    assert doc.page_content == "Live abstract content about neural network designs."
    assert doc.metadata["arxiv_id"] == "2301.00002"
    assert doc.metadata["title"] == "Live Paper Title"
    assert doc.metadata["source"] == "https://arxiv.org/pdf/2301.00002.pdf"
    assert doc.metadata["search_mode"] == "live_arxiv"


@pytest.mark.asyncio
async def test_google_search_tool(monkeypatch):
    """Test the google_search tool using DuckDuckGo mock."""
    mock_results = [
        {
            "title": "Mock Web Page Title",
            "href": "https://example.com/mock-page",
            "body": "This is a mock snippet from the web search results."
        }
    ]

    class MockDDGS:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
        def text(self, query, max_results=3):
            return mock_results

    monkeypatch.setattr("duckduckgo_search.DDGS", MockDDGS)

    result = await google_search.ainvoke({"query": "test query"})

    assert isinstance(result, list)
    assert len(result) == 1
    doc = result[0]
    assert doc.page_content == "This is a mock snippet from the web search results."
    assert doc.metadata["arxiv_id"] == "web"
    assert doc.metadata["title"] == "Mock Web Page Title"
    assert doc.metadata["source"] == "https://example.com/mock-page"
    assert doc.metadata["search_mode"] == "web_search"

