import asyncio
import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_core.tools import tool
from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.opensearch.client import OpenSearchClient

logger = logging.getLogger(__name__)


def create_retriever_tool(
    opensearch_client: OpenSearchClient,
    embeddings_client: JinaEmbeddingsClient,
    top_k: int = 3,
    use_hybrid: bool = True,
    tenant_id: Optional[str] = None,
):
    """Create a retriever tool that wraps OpenSearch service.

    :param opensearch_client: Existing OpenSearch service
    :param embeddings_client: Existing Jina embeddings service
    :param top_k: Number of chunks to retrieve
    :param use_hybrid: Use hybrid search (BM25 + vector)
    :param tenant_id: Optional tenant isolation identifier
    :returns: LangChain tool for retrieving papers
    """

    @tool
    async def retrieve_papers(query: str) -> list[Document]:
        """Search and return relevant arXiv research papers.

        Use this tool when the user asks about:
        - Machine learning concepts or techniques
        - Deep learning architectures
        - Natural language processing
        - Computer vision methods
        - AI research topics
        - Specific algorithms or models

        :param query: The search query describing what papers to find
        :returns: List of relevant paper excerpts with metadata
        """
        logger.info(f"Retrieving papers for query: {query[:100]}...")
        logger.debug(f"Search mode: {'hybrid' if use_hybrid else 'bm25'}, top_k: {top_k}")

        # Generate query embedding
        logger.debug("Generating query embedding")
        query_embedding = await embeddings_client.embed_query(query)
        if query_embedding is not None:
            logger.debug(f"Generated embedding with {len(query_embedding)} dimensions")
        else:
            logger.debug("No query embedding generated (will fallback to BM25)")

        # Search using OpenSearch
        logger.debug("Searching OpenSearch")
        search_results = await opensearch_client.search_unified(
            query=query,
            query_embedding=query_embedding,
            size=top_k,
            use_hybrid=use_hybrid,
            tenant_id=tenant_id,
        )

        # Convert SearchHit to LangChain Document
        documents = []
        hits = search_results.get("hits", [])
        logger.info(f"Found {len(hits)} documents from OpenSearch")

        seen_parents = set()
        for hit in hits:
            parent_id = hit.get("parent_id")
            parent_text = hit.get("parent_text")

            if parent_text:
                identifier = parent_id if parent_id else parent_text
                if identifier in seen_parents:
                    continue
                seen_parents.add(identifier)
                page_content = parent_text
            else:
                page_content = hit["chunk_text"]

            doc = Document(
                page_content=page_content,
                metadata={
                    "arxiv_id": hit["arxiv_id"],
                    "title": hit.get("title", ""),
                    "authors": hit.get("authors", ""),
                    "score": hit.get("score", 0.0),
                    "source": f"https://arxiv.org/pdf/{hit['arxiv_id']}.pdf",
                    "section": hit.get("section_name", ""),
                    "search_mode": "hybrid" if use_hybrid else "bm25",
                    "top_k": top_k,
                },
            )
            documents.append(doc)

        logger.debug(f"Converted {len(documents)} hits to LangChain Documents")
        logger.info(f"✓ Retrieved {len(documents)} papers successfully")

        return documents

    return retrieve_papers


from src.services.arxiv.client import ArxivClient


def create_arxiv_search_tool(
    arxiv_client: ArxivClient,
    top_k: int = 3,
):
    """Create a live arXiv search tool.

    :param arxiv_client: Existing arXiv API client
    :param top_k: Maximum results to retrieve
    :returns: LangChain tool for searching live arXiv API
    """

    @tool
    async def search_arxiv(query: str) -> list[Document]:
        """Search the live arXiv API for CS/AI/ML research papers.

        Use this tool when:
        - The user explicitly requests searching the web, live arXiv API, or online search
        - No relevant papers were found in the local database and a broader search is needed
        - You need to retrieve the absolute latest research papers on a topic

        :param query: The search keywords or query
        :returns: List of relevant papers from arXiv API with abstracts
        """
        logger.info(f"Searching live arXiv API for query: {query}")
        try:
            # Query arXiv using custom query search
            # We target titles and abstracts, but using "all:{query}" queries everything
            search_query = f"all:{query} AND (cat:cs.AI OR cat:cs.LG OR cat:cs.CV OR cat:cs.NE OR cat:cs.CL)"
            papers = await arxiv_client.fetch_papers_with_query(
                search_query=search_query,
                max_results=top_k,
                sort_by="relevance",
            )

            documents = []
            for paper in papers:
                doc = Document(
                    page_content=paper.abstract,
                    metadata={
                        "arxiv_id": paper.arxiv_id,
                        "title": paper.title,
                        "authors": paper.authors,
                        "score": 1.0,
                        "source": paper.pdf_url or f"https://arxiv.org/pdf/{paper.arxiv_id}.pdf",
                        "section": "Abstract",
                        "search_mode": "live_arxiv",
                        "top_k": top_k,
                    },
                )
                documents.append(doc)

            logger.info(f"✓ Live arXiv search found {len(documents)} papers")
            return documents
        except Exception as e:
            logger.error(f"Live arXiv search failed: {e}")
            return []

    return search_arxiv


@tool
async def google_search(query: str) -> list[Document]:
    """Search Google/Web for research papers and articles.

    Use this tool when:
    - The query is classified as a web search or require online research.
    - You need to look up concepts, definitions, or recent papers from the web.

    :param query: The search query to look up on the web.
    :returns: List of documents representing search results with clean titles and source URLs.
    """
    logger.info(f"Executing web/google search for query: {query}")
    try:
        from duckduckgo_search import DDGS

        def _ddg_search(q: str) -> list:
            with DDGS() as ddgs:
                return list(ddgs.text(q, max_results=3))

        results = await asyncio.to_thread(_ddg_search, query)

        documents = []
        for res in results:
            doc = Document(
                page_content=res.get("body", ""),
                metadata={
                    "arxiv_id": "web",  # Fallback for UI matching
                    "title": res.get("title", "Untitled Web Result"),
                    "authors": ["Web Search"],
                    "score": 1.0,
                    "source": res.get("href", "#"),
                    "section": "Web Snippet",
                    "search_mode": "web_search",
                    "top_k": 3,
                },
            )
            documents.append(doc)

        logger.info(f"✓ Web search completed. Found {len(documents)} results")
        return documents
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return []
