from typing import TYPE_CHECKING, Annotated, Generator, Optional

if TYPE_CHECKING:
    from fastapi import Depends, Request
    from sqlalchemy.orm import Session
else:
    try:
        from fastapi import Depends, Request
        from sqlalchemy.orm import Session
    except ImportError:
        pass

from src.config import Settings
from src.db.interfaces.base import BaseDatabase
from src.services.agents.agentic_rag import AgenticRAGService
from src.services.agents.factory import make_agentic_rag_service
from src.services.arxiv.client import ArxivClient
from src.services.auth.api_key_service import APIKeyMetadata, APIKeyService, require_api_key
from src.services.cache.client import CacheClient
from src.services.cache.semantic_cache import SemanticCache
from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.langfuse.client import LangfuseTracer
from src.services.ollama.client import OllamaClient
from src.services.opensearch.client import OpenSearchClient
from src.services.pdf_parser.parser import PDFParserService
from src.services.reranker.client import RerankerClient
from src.services.telegram.bot import TelegramBot
from src.services.tenant import TenantContext, require_tenant
from src.services.web_search import WebSearchService


def get_request_settings(request: Request) -> Settings:
    """Get settings from the request state."""
    return request.app.state.settings


def get_database(request: Request) -> BaseDatabase:
    """Get database from the request state."""
    return request.app.state.database


def get_db_session(database: Annotated[BaseDatabase, Depends(get_database)]) -> Generator[Session, None, None]:
    """Get database session dependency."""
    with database.get_session() as session:
        yield session


def get_opensearch_client(request: Request) -> OpenSearchClient:
    """Get OpenSearch client from the request state."""
    return request.app.state.opensearch_client


def get_arxiv_client(request: Request) -> ArxivClient:
    """Get arXiv client from the request state."""
    return request.app.state.arxiv_client


def get_pdf_parser(request: Request) -> PDFParserService:
    """Get PDF parser service from the request state."""
    return request.app.state.pdf_parser


def get_embeddings_service(request: Request) -> JinaEmbeddingsClient:
    """Get embeddings service from the request state."""
    return request.app.state.embeddings_service


def get_ollama_client(request: Request) -> OllamaClient:
    """Get Ollama client from the request state."""
    return request.app.state.ollama_client


def get_langfuse_tracer(request: Request) -> LangfuseTracer:
    """Get Langfuse tracer from the request state."""
    return request.app.state.langfuse_tracer


def get_cache_client(request: Request) -> CacheClient | None:
    """Get cache client from the request state."""
    return getattr(request.app.state, "cache_client", None)


def get_semantic_cache(request: Request) -> Optional[SemanticCache]:
    """Get semantic cache client from the request state."""
    return getattr(request.app.state, "semantic_cache", None)


def get_telegram_service(request: Request) -> Optional[TelegramBot]:
    """Get Telegram service from the request state."""
    return getattr(request.app.state, "telegram_service", None)


# Dependency annotations
SettingsDep = Annotated[Settings, Depends(get_request_settings)]
DatabaseDep = Annotated[BaseDatabase, Depends(get_database)]
SessionDep = Annotated[Session, Depends(get_db_session)]
OpenSearchDep = Annotated[OpenSearchClient, Depends(get_opensearch_client)]
ArxivDep = Annotated[ArxivClient, Depends(get_arxiv_client)]
PDFParserDep = Annotated[PDFParserService, Depends(get_pdf_parser)]
EmbeddingsDep = Annotated[JinaEmbeddingsClient, Depends(get_embeddings_service)]
OllamaDep = Annotated[OllamaClient, Depends(get_ollama_client)]
LangfuseDep = Annotated[LangfuseTracer, Depends(get_langfuse_tracer)]
CacheDep = Annotated[CacheClient | None, Depends(get_cache_client)]
SemanticCacheDep = Annotated[Optional[SemanticCache], Depends(get_semantic_cache)]
TelegramDep = Annotated[Optional[TelegramBot], Depends(get_telegram_service)]
APIKeyDep = Annotated[APIKeyMetadata, Depends(require_api_key)]
TenantDep = Annotated[TenantContext, Depends(require_tenant)]


def get_api_key_service(request: Request) -> Optional[APIKeyService]:
    """Get API key service from the request state."""
    return getattr(request.app.state, "api_key_service", None)


APIKeyServiceDep = Annotated[Optional[APIKeyService], Depends(get_api_key_service)]


def get_reranker_client(request: Request) -> Optional[RerankerClient]:
    """Get reranker client from the request state."""
    return getattr(request.app.state, "reranker_client", None)


RerankerDep = Annotated[Optional[RerankerClient], Depends(get_reranker_client)]


def get_agentic_rag_service(request: Request) -> AgenticRAGService:
    """Get cached agentic RAG service from app state (avoids rebuilding graph per request).

    The graph is expensive to build (involves LLM node compilation), so we
    cache it at app.state.agentic_rag_service and rebuild only when the
    service is first needed or explicitly reset.
    """
    service = getattr(request.app.state, "agentic_rag_service", None)
    if service is None:
        settings = request.app.state.settings
        service = make_agentic_rag_service(
            opensearch_client=request.app.state.opensearch_client,
            ollama_client=request.app.state.ollama_client,
            embeddings_client=request.app.state.embeddings_service,
            langfuse_tracer=request.app.state.langfuse_tracer,
            reranker_client=getattr(request.app.state, "reranker_client", None),
            model=settings.ollama_model,
        )
        request.app.state.agentic_rag_service = service
    return service


AgenticRAGDep = Annotated[AgenticRAGService, Depends(get_agentic_rag_service)]


def get_web_search_service(request: Request) -> WebSearchService:
    """Get web search service from the request state."""
    return request.app.state.web_search_service


WebSearchDep = Annotated[WebSearchService, Depends(get_web_search_service)]


def get_conversation_memory_service(request: Request):
    """Get conversation memory service from the request state."""
    from src.services.conversation_memory import ConversationMemoryService
    return request.app.state.conversation_memory_service


ConversationMemoryDep = Annotated[
    any, Depends(get_conversation_memory_service)
]


