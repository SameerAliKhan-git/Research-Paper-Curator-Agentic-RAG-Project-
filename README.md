# The Mother of AI Project
## Production RAG Systems Hackathon

<div align="center">
  <h3>A Hands-On Hackathon for Building Production RAG Systems</h3>
  <p>Learn to build modern AI systems from the ground up through two progressive phases</p>
  <p>Master the most in-demand AI engineering skill: <strong>RAG (Retrieval-Augmented Generation)</strong></p>
</div>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/FastAPI-0.115+-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/OpenSearch-2.19-orange.svg" alt="OpenSearch">
  <img src="https://img.shields.io/badge/Docker-Compose-blue.svg" alt="Docker">
</p>

<br>

<p align="center">
  <a href="#-about-this-hackathon">
    <img src="static/mother_of_ai_project_rag_architecture.gif" alt="RAG Architecture" width="700">
  </a>
</p>

## About This Hackathon

This is a **4-day hackathon** (2 days per phase) where you'll build a complete, production-grade RAG system. Each phase is self-contained with its own Docker Compose, dependencies, and documentation.

| Phase | Duration | Focus | What You'll Build |
|-------|----------|-------|-------------------|
| [**Phase 1**](phase1/) | 2 days | Basic RAG | End-to-end RAG pipeline with document upload, vector search, and streaming LLM responses |
| [**Phase 2**](phase2/) | 2 days | Advanced/Production RAG | Hybrid search, re-ranking, caching, observability, and automated ingestion pipelines |

> **Progression:** Phase 1 gives you a working RAG system. Phase 2 makes it production-grade with the tools and patterns used by real companies.

---

## Prerequisites

- **Docker Desktop** (with Docker Compose)
- **Python 3.12+**
- **UV Package Manager** ([Install Guide](https://docs.astral.sh/uv/getting-started/installation/))
- **8GB+ RAM** and **20GB+ free disk space**

---

## Quick Start

### Phase 1 — Basic RAG (Start Here)

```bash
cd phase1
cp .env.example .env
uv sync
docker compose up --build -d

# Verify
curl http://localhost:8000/health
# Open Gradio UI: http://localhost:7861
```

Phase 1 runs **4 services**: FastAPI, Gradio, OpenSearch, Ollama.

See the full [Phase 1 README](phase1/README.md) for details.

### Phase 2 — Advanced/Production RAG

```bash
cd phase2
cp .env.example .env
uv sync
docker compose up --build -d

# Verify
curl http://localhost:8000/health
# Open Gradio UI: http://localhost:7861
```

Phase 2 runs **14 services** including PostgreSQL, Redis, Airflow, and the full Langfuse observability stack.

See the full [Phase 2 README](phase2/README.md) for details.

---

## Repository Structure

```
production-agentic-rag-course/
├── README.md                # This file
├── phase1/                  # Basic RAG (fully independent)
│   ├── README.md            # Phase 1 setup guide
│   ├── compose.yml          # 4 Docker services
│   ├── pyproject.toml       # Phase 1 dependencies
│   ├── src/                 # Application source code
│   ├── tests/               # Test suite
│   └── notebooks/           # Phase 1 learning notebooks
├── phase2/                  # Advanced/Production RAG (self-contained)
│   ├── README.md            # Phase 2 setup guide
│   ├── compose.yml          # 14 Docker services
│   ├── pyproject.toml       # Phase 2 dependencies
│   ├── airflow/             # Apache Airflow ingestion pipeline
│   ├── src/                 # Application source code
│   ├── tests/               # Test suite
│   └── notebooks/           # Phase 2 learning notebooks
├── archive/                 # Original weekly notebooks (reference)
│   └── notebooks/
│       ├── week1/ ... week7/
└── static/                  # Architecture diagrams
```

---

## Phase Comparison

| Feature | Phase 1 | Phase 2 |
|---------|---------|---------|
| **Search** | Vector-only (k-NN) | Hybrid (BM25 + Vector with RRF) |
| **Embeddings** | Multi-provider (local/Jina/OpenAI) | Multi-provider (local/Jina/OpenAI) |
| **LLM** | Ollama (local) | Ollama (local) |
| **Storage** | OpenSearch | OpenSearch + PostgreSQL |
| **Caching** | None | Redis with TTL |
| **Re-ranking** | None | Cohere cross-encoder |
| **Observability** | None | Langfuse tracing |
| **Logging** | Basic | JSON structured logging |
| **Ingestion** | Manual upload (Gradio) | Manual upload + Airflow pipelines |
| **Docker services** | 4 | 14 |

---

## Technology Stack

| Component | Technology | Used In |
|-----------|-----------|---------|
| **API Framework** | FastAPI | Both |
| **Vector/Search DB** | OpenSearch 2.19 | Both |
| **LLM Server** | Ollama | Both |
| **UI** | Gradio | Both |
| **Embeddings** | sentence-transformers / Jina / OpenAI | Both |
| **PDF Parsing** | Docling | Both |
| **Metadata DB** | PostgreSQL 16 | Phase 2 |
| **Cache** | Redis 7 | Phase 2 |
| **Orchestration** | Apache Airflow | Phase 2 |
| **Observability** | Langfuse v3 | Phase 2 |
| **Re-ranking** | Cohere Rerank | Phase 2 |
| **Dev Tools** | uv, ruff, pytest, Docker Compose | Both |

---

## Suggested Pacing

### Phase 1 (2 Days)
- **Day 1**: Infrastructure setup (Docker, OpenSearch, Ollama), PDF parsing, text preprocessing, chunking, embeddings
- **Day 2**: Vector search, RAG pipeline, Gradio UI with upload, streaming responses, dev tooling

### Phase 2 (2 Days)
- **Day 1**: Hybrid search with RRF, PostgreSQL metadata, Airflow ingestion pipeline, advanced FastAPI patterns
- **Day 2**: Redis caching, Cohere re-ranking, Langfuse observability, JSON structured logging

---

## Blog Posts (Reference)

These blog posts from the original weekly course provide deeper context:

| Topic | Blog Post |
|-------|-----------|
| Infrastructure | [The Infrastructure That Powers RAG Systems](https://jamwithai.substack.com/p/the-infrastructure-that-powers-rag) |
| Data Ingestion | [Building Data Ingestion Pipelines for RAG](https://jamwithai.substack.com/p/bringing-your-rag-system-to-life) |
| BM25 Search | [The Search Foundation Every RAG System Needs](https://jamwithai.substack.com/p/the-search-foundation-every-rag-system) |
| Hybrid Search | [The Chunking Strategy That Makes Hybrid Search Work](https://jamwithai.substack.com/p/chunking-strategies-and-hybrid-rag) |
| Complete RAG | [The Complete RAG System](https://jamwithai.substack.com/p/the-complete-rag-system) |
| Monitoring & Caching | [Production-ready RAG: Monitoring & Caching](https://jamwithai.substack.com/p/production-ready-rag-monitoring-and) |

---

## Cost Structure

**This hackathon is free!** Everything runs locally by default.

- **Local development**: $0 (local embeddings + Ollama LLM)
- **Optional cloud APIs**: ~$2-5 (Jina/OpenAI embeddings, Cohere re-ranking)

---

## Troubleshooting

- **Services not starting?** Wait 2-3 minutes for health checks, then run `docker compose logs`
- **Port conflicts?** Ensure ports 8000, 7861, 9200, 11434 are free (Phase 1) — Phase 2 also needs 5432, 6379, 8080, 3001
- **Memory issues?** Increase Docker Desktop memory (8GB+ recommended)
- **Complete reset**: `docker compose down --volumes && docker compose up --build -d`

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=jamwithai/production-agentic-rag-course&type=Date)](https://star-history.com/#jamwithai/production-agentic-rag-course&Date)

---

<div align="center">
  <p><strong>Built with love by Jam With AI</strong></p>
</div>
