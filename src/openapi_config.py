from fastapi import FastAPI


def configure_openapi(app: FastAPI) -> None:
    """Customize the FastAPI OpenAPI schema with tags, docs, and metadata."""
    original_openapi = app.openapi

    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        schema = original_openapi()
        schema["info"]["title"] = "arXiv Paper Curator API"
        schema["info"]["description"] = (
            "Production-Grade Agentic arXiv Paper Curator RAG System. "
            "Provides endpoints for paper ingestion, hybrid search, RAG question answering, "
            "and administrative operations."
        )
        schema["info"]["version"] = "0.1.0"
        schema["info"]["contact"] = {
            "name": "API Support",
            "email": "support@rag-api.dev",
            "url": "https://github.com/example/rag-api",
        }
        schema["info"]["license"] = {
            "name": "MIT License",
            "url": "https://opensource.org/licenses/MIT",
        }
        schema["tags"] = [
            {"name": "ping", "description": "Health check and readiness probes"},
            {"name": "papers", "description": "Paper listing, ingestion, and synchronization"},
            {"name": "search", "description": "Hybrid BM25 + vector search across papers"},
            {"name": "ask", "description": "RAG question answering with LLM"},
            {"name": "agentic", "description": "Agentic RAG with intelligent retrieval planning"},
            {"name": "summarize", "description": "Paper summarization using LLM"},
            {"name": "related", "description": "Find related papers by vector similarity"},
            {"name": "bulk", "description": "Bulk paper import operations"},
            {"name": "export", "description": "Citation and data export"},
            {"name": "admin", "description": "Administrative dashboard and system statistics"},
            {"name": "websocket", "description": "WebSocket streaming for real-time RAG responses"},
            {"name": "operations", "description": "Metrics and observability endpoints"},
        ]
        schema["externalDocs"] = {
            "description": "Project Documentation",
            "url": "https://github.com/example/rag-api/wiki",
        }
        app.openapi_schema = schema
        return schema

    app.openapi = custom_openapi
