import asyncio
import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.config import get_settings
from src.database import init_database
from src.graphql.schema import schema as graphql_schema
from src.logging_config import LogContextMiddleware, setup_logging
from src.middlewares import (
    ExceptionCaptureMiddleware,
    RateLimitMiddleware,
    RequestBodySizeLimitMiddleware,
    RequestLoggingMiddleware,
)
from src.routers import (
    admin,
    agentic_ask,
    ask,
    bulk,
    citations,
    export,
    hybrid_search,
    literature,
    papers,
    ping,
    related,
    review,
    summarize,
    trends,
    users,
    collections,
    annotations,
    sync,
    visual_search,
    evaluation,
)
from src.routers.ask import ask_router, stream_router
from src.routers.ws import ws_router
from src.services.arxiv.factory import make_arxiv_client
from src.services.auth.factory import make_api_key_service
from src.services.cache.factory import make_cache_client
from src.services.cache.semantic_factory import make_semantic_cache
from src.services.cost_tracker import CostMiddleware, cost_tracker
from src.services.embeddings.factory import make_embeddings_service
from src.services.langfuse.factory import make_langfuse_tracer
from src.services.metrics import PrometheusMiddleware, get_metrics_endpoint, metrics
from src.services.ollama.factory import make_ollama_client
from src.services.opensearch.factory import make_opensearch_client
from src.services.pdf_parser.factory import make_pdf_parser_service
from src.services.reranker.factory import make_reranker_service
from src.services.telegram.factory import make_telegram_service
from src.services.tenant import TenantMiddleware
from src.services.tracing import TracingMiddleware, setup_opentelemetry
from src.services.web_search import WebSearchService

settings = get_settings()

# Setup structured logging
setup_logging(environment=settings.environment)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan for the API.
    """
    logger.info("Starting RAG API...")

    # Setup OpenTelemetry
    setup_opentelemetry(
        service_name=settings.service_name,
        environment=settings.environment,
        otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
    )

    app.state.settings = settings

    database = init_database()
    app.state.database = database
    logger.info("Database connected")

    # Initialize search service
    opensearch_client = make_opensearch_client()
    app.state.opensearch_client = opensearch_client

    # Verify OpenSearch connectivity and create index if needed
    if opensearch_client.health_check():
        logger.info("OpenSearch connected successfully")

        # Setup hybrid index (supports all search types)
        setup_results = opensearch_client.setup_indices(force=False)
        if setup_results.get("hybrid_index"):
            logger.info("Hybrid index created")
        else:
            logger.info("Hybrid index already exists")

        # Get simple statistics
        try:
            stats = opensearch_client.client.count(index=opensearch_client.index_name)
            logger.info(f"OpenSearch ready: {stats['count']} documents indexed")
        except Exception:
            logger.info("OpenSearch index ready (stats unavailable)")
    else:
        logger.warning("OpenSearch connection failed - search features will be limited")

    # Initialize other services (kept for future endpoints and notebook demos)
    app.state.arxiv_client = make_arxiv_client()
    app.state.pdf_parser = make_pdf_parser_service()
    app.state.embeddings_service = make_embeddings_service()
    app.state.ollama_client = make_ollama_client()
    app.state.langfuse_tracer = make_langfuse_tracer()
    app.state.api_key_service = await make_api_key_service(settings)
    if app.state.api_key_service:
        try:
            await app.state.api_key_service.create_key(
                raw_key="dev-test-key-999",
                user_id="developer@local",
                tier="premium",
                rate_limit=99999,
                daily_quota=999999,
                tenants=["default", "research-tenant"]
            )
            logger.info("Seeded default dev-test-key-999 API Key in Redis")
        except Exception as e:
            logger.warning(f"Failed to seed dev-test-key-999: {e}")
    app.state.reranker_client = make_reranker_service()


    # Cache clients are now async - initialize with await
    try:
        app.state.cache_client = await make_cache_client(settings)
    except Exception:
        app.state.cache_client = None

    try:
        app.state.semantic_cache = await make_semantic_cache(settings)
    except Exception:
        app.state.semantic_cache = None

    # Initialize WebSearchService
    redis_client = getattr(app.state.cache_client, "redis", None) if app.state.cache_client else None
    app.state.web_search_service = WebSearchService(redis_client=redis_client)

    # Initialize ConversationMemoryService
    from src.services.conversation_memory import ConversationMemoryService
    app.state.conversation_memory_service = ConversationMemoryService(settings, redis_client=redis_client)

    # Initialize cost tracker with Redis persistence
    cache_client = getattr(app.state, "cache_client", None)
    if cache_client is not None:
        cost_tracker._redis = cache_client.redis
        await cost_tracker.load_from_redis()
        logger.info("Cost tracker initialized with Redis persistence")
    else:
        logger.info("Cost tracker running in memory-only mode")

    logger.info(
        "Services initialized: arXiv API client, PDF parser, OpenSearch, Embeddings, Ollama, Langfuse, Cache, "
        "Semantic Cache, API Key Service, Reranker"
    )

    # Initialize Telegram bot (Week 7)
    telegram_service = make_telegram_service(
        opensearch_client=app.state.opensearch_client,
        embeddings_client=app.state.embeddings_service,
        ollama_client=app.state.ollama_client,
        cache_client=app.state.cache_client,
        langfuse_tracer=app.state.langfuse_tracer,
    )

    if telegram_service:
        app.state.telegram_service = telegram_service
        try:
            await telegram_service.start()
            app.state.telegram_started = True
            logger.info("Telegram bot started successfully")
        except Exception as e:
            app.state.telegram_started = False
            logger.error("Failed to start Telegram bot")
    else:
        app.state.telegram_started = False
        logger.info("Telegram bot not configured - skipping initialization")

    # Background task for semantic cache cleanup
    async def _cache_cleanup_loop():
        while True:
            await asyncio.sleep(3600)  # every hour
            if getattr(app.state, "semantic_cache", None):
                try:
                    removed = await app.state.semantic_cache.cleanup_orphans()
                    if removed:
                        logger.info(f"Cache cleanup: removed {removed} orphaned entries")
                except Exception as e:
                    logger.warning(f"Cache cleanup task error: {e}")

    app.state._cache_cleanup_task = asyncio.create_task(_cache_cleanup_loop())

    logger.info("API ready")
    yield

    # Cancel background tasks
    if hasattr(app.state, "_cache_cleanup_task"):
        app.state._cache_cleanup_task.cancel()
        try:
            await app.state._cache_cleanup_task
        except asyncio.CancelledError:
            pass

    # Cleanup
    if getattr(app.state, "telegram_started", False) and hasattr(app.state, "telegram_service") and app.state.telegram_service:
        try:
            await app.state.telegram_service.stop()
            logger.info("Telegram bot stopped")
        except Exception:
            logger.warning("Telegram bot stop error (non-fatal)")

    # Cleanup HTTP clients
    if hasattr(app.state, "embeddings_service") and app.state.embeddings_service:
        try:
            await app.state.embeddings_service.close()
        except Exception:
            pass

    if hasattr(app.state, "reranker_client") and app.state.reranker_client:
        try:
            await app.state.reranker_client.close()
        except Exception:
            pass

    if hasattr(app.state, "ollama_client") and app.state.ollama_client:
        try:
            await app.state.ollama_client.close()
        except Exception:
            pass

    # Flush Langfuse traces before shutdown
    if hasattr(app.state, "langfuse_tracer") and app.state.langfuse_tracer:
        try:
            app.state.langfuse_tracer.shutdown()
        except Exception:
            pass

    # Close async Redis cache connections
    if hasattr(app.state, "cache_client") and app.state.cache_client:
        try:
            await app.state.cache_client.redis.aclose()
        except Exception:
            pass

    if hasattr(app.state, "semantic_cache") and app.state.semantic_cache:
        try:
            await app.state.semantic_cache.redis.aclose()
        except Exception:
            pass

    # Close async Redis auth connection
    if hasattr(app.state, "api_key_service") and app.state.api_key_service:
        try:
            await app.state.api_key_service.redis.aclose()
        except Exception:
            pass

    database.teardown()
    logger.info("API shutdown complete")


app = FastAPI(
    title="arXiv Paper Curator API",
    description="Personal arXiv CS.AI paper curator with RAG capabilities",
    version=os.getenv("APP_VERSION", "0.1.0"),
    lifespan=lifespan,
)

# Apply OpenAPI customization
from src.openapi_config import configure_openapi

configure_openapi(app)

# Include routers
app.include_router(ping.router, prefix="/api/v1")  # Health check endpoint
app.include_router(hybrid_search.router, prefix="/api/v1")  # Search chunks with BM25/hybrid
app.include_router(ask_router, prefix="/api/v1")  # RAG question answering with LLM
app.include_router(stream_router, prefix="/api/v1")  # Streaming RAG responses
app.include_router(agentic_ask.router)  # Agentic RAG with intelligent retrieval
app.include_router(papers.router, prefix="/api/v1")  # List indexed papers from DB
app.include_router(related.router, prefix="/api/v1")  # Find related papers by vector similarity
app.include_router(summarize.router, prefix="/api/v1")  # Summarize papers with LLM
app.include_router(export.router, prefix="/api/v1")  # Export citations
app.include_router(ws_router, prefix="/api/v1")  # WebSocket streaming RAG
app.include_router(bulk.router, prefix="/api/v1")  # Bulk paper import
app.include_router(admin.router, prefix="/api/v1")  # Admin dashboard
app.include_router(citations.router, prefix="/api/v1")  # Citation network graph
app.include_router(trends.router, prefix="/api/v1")  # Research trends over time
app.include_router(review.router, prefix="/api/v1")  # Literature review generation
app.include_router(literature.router, prefix="/api/v1")  # LaTeX related-work Comparative Synthesis
app.include_router(users.router, prefix="/api/v1")  # User authentication
app.include_router(collections.router, prefix="/api/v1")  # Collections API
app.include_router(annotations.router, prefix="/api/v1")  # Annotations API
app.include_router(sync.router, prefix="/api/v1")  # Obsidian & Notion Sync API
app.include_router(visual_search.router, prefix="/api/v1")  # ColPali Visual Search API
app.include_router(evaluation.router, prefix="/api/v1")  # RAGAS Evaluation Suite
from src.routers import recommendations
app.include_router(recommendations.router, prefix="/api/v1")  # Recommendations API


# GraphQL endpoint
from strawberry.fastapi import GraphQLRouter

graphql_router = GraphQLRouter(graphql_schema)
app.include_router(graphql_router, prefix="/graphql")

# Prometheus Metrics
app.add_api_route("/metrics", get_metrics_endpoint(), methods=["GET"], tags=["monitoring"])

# CORS middleware - restrict origins for production
if settings.environment == "production":
    cors_origins_raw = os.getenv("CORS_ALLOWED_ORIGINS", "")
    ALLOWED_ORIGINS = [o.strip() for o in cors_origins_raw.split(",") if o.strip()]
    if not ALLOWED_ORIGINS:
        logger.warning("CORS_ALLOWED_ORIGINS is empty in production — all cross-origin requests will be blocked")
else:
    ALLOWED_ORIGINS = os.getenv(
        "CORS_ALLOWED_ORIGINS",
        "http://localhost:8000,http://localhost:3000",
    ).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Observability middleware
app.add_middleware(PrometheusMiddleware, metrics_collector=metrics)
app.add_middleware(CostMiddleware)
app.add_middleware(TracingMiddleware)
app.add_middleware(TenantMiddleware)

# Request logging + correlation ID
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(LogContextMiddleware)

# Error handling
app.add_middleware(ExceptionCaptureMiddleware)

# Rate limiting (per API key)
app.add_middleware(RateLimitMiddleware, default_rate_limit=settings.default_rate_limit, window_seconds=60)

# Request body size limit (outermost - rejects oversized payloads before they hit anything else)
max_size_bytes = settings.max_request_size_mb * 1024 * 1024
app.add_middleware(RequestBodySizeLimitMiddleware, max_size_bytes=max_size_bytes)

# Metrics endpoint
app.add_api_route("/metrics", get_metrics_endpoint(), tags=["operations"])

# Serve UI static files and mount assets
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def serve_index():
    """Serve the production Replicate-themed web interface."""
    return FileResponse("static/index.html")


if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="0.0.0.0")
