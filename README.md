# arXiv Paper Curator: Production-Grade Agentic RAG System

<div align="center">
  <h3>Enterprise Search & Cognitive Synthesis for Academic Literature</h3>
  <p>High-throughput data extraction, hybrid retrieval, and multi-agent reasoning orchestrations</p>
</div>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/FastAPI-0.115+-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/OpenSearch-2.19-orange.svg" alt="OpenSearch">
  <img src="https://img.shields.io/badge/Docker-Compose-blue.svg" alt="Docker">
  <img src="https://img.shields.io/badge/Orchestrator-Apache_Airflow_3.0-darkblue.svg" alt="Airflow">
  <img src="https://img.shields.io/badge/Status-Production_Ready-brightgreen.svg" alt="Status">
</p>

</br>

<p align="center">
  <img src="static/mother_of_ai_project_rag_architecture.gif" alt="RAG Architecture" width="700">
</p>

---

## 🔬 System Overview

The **arXiv Paper Curator** is a production-grade, enterprise-ready **Agentic Retrieval-Augmented Generation (RAG)** platform designed to automate literature search, extract structured knowledge from academic PDFs, and provide high-accuracy, grounded question answering. 

Unlike basic vector search systems, this application uses a hybrid index (combining keyword and vector search) and a state-based multi-agent routing loop to eliminate hallucination, evaluate retrieval quality dynamically, and refine query definitions in real-time.

### ⚡ Key Capabilities

- **Asynchronous Processing Pipeline**: Leverages thread-isolated Docling parsers inside Apache Airflow DAGs to fetch metadata and parse complex table/text layouts of academic PDFs, loading them into PostgreSQL and OpenSearch indexes.
- **RRF Hybrid Search**: Merges OpenSearch BM25 keyword matching with high-dimensionality vector search (Jina AI embeddings) using Reciprocal Rank Fusion (RRF) for maximized recall and precision.
- **Adaptive Agentic Reasoning**: Orchestrates LangGraph state machines containing validation guardrails, semantic document grading, automatic query rewriting, and graceful out-of-scope rejections.
- **Production Observability**: Traces execution latency, token costs, and node transitions in Langfuse, supporting online user ratings feedback.
- **Sub-Millisecond Cache**: Employs Redis exact-match query caching to bypass LLM calls for repeating queries, achieving up to 400x speedups.
- **Interfaces**:
  - **Console Dashboard** (`http://localhost:8000/`): Editorial, zine-style developer console based on Replicate's design tokens (warm cream canvas, hot orange brand accents, dark code wells, and interactive timeline streaming).
  - **Gradio Console** (`http://localhost:7861/`): Streamlined alternative chat window for rapid local prototyping.
  - **Telegram Bot Entrypoint**: Integrates mobile interaction for on-the-go research lookups.

---

## 🏗️ System Architecture

### Multi-Agent LangGraph Reasoning Loop
```
                                  [ User Query ]
                                        │
                                        ▼
                              ┌───────────────────┐
                              │  Guardrail Node   │ ──(Violates domain)──► [ Helpfully Reject ]
                              └───────────────────┘
                                        │ (Passes)
                                        ▼
                              ┌───────────────────┐
                              │   Retrieve Node   │
                              └───────────────────┘
                                        │
                                        ▼
                              ┌───────────────────┐
                              │  Grade Documents  │
                              └───────────────────┘
                                   /         \
                       (All Irrelevant)     (At least 1 Relevant)
                                 /             \
                                ▼               ▼
                      ┌────────────────┐      ┌───────────────────┐
                      │  Rewrite Query │      │  Generate Answer  │
                      └────────────────┘      └───────────────────┘
                                │                       │
                     [ Retry Retrieve Node ]     [ Stream / Respond ]
```

### Ingestion Flow Diagram
<div align="center">
  <img src="static/week2_data_ingestion_flow.png" alt="Ingestion Pipeline" width="800">
</div>

---

## 🚀 Quick Start

### 📋 Prerequisites
- **Docker Desktop** (with Docker Compose)
- **Python 3.12+**
- **UV Package Manager** ([Install Guide](https://docs.astral.sh/uv/getting-started/installation/))
- **8GB+ RAM** and **20GB+ free disk space**

### ⚡ Installation & Launch

```bash
# 1. Clone the repository
git clone <repository-url>
cd arxiv-paper-curator

# 2. Configure environment keys
cp .env.example .env
# Open .env and insert:
# - JINA_API_KEY (For Vector embeddings)
# - TELEGRAM__BOT_TOKEN (For Telegram bot, optional)
# - LANGFUSE__PUBLIC_KEY & LANGFUSE__SECRET_KEY (For tracing, optional)

# 3. Synchronize virtual environment dependencies
uv sync

# 4. Start Docker Compose infrastructure services
docker compose up --build -d

# 5. Verify local health checks
curl http://localhost:8000/api/v1/health
```

---

## 📡 Port & Services Allocation

Once services have successfully initialized, you can access the following dashboards:

| Service Dashboard | URL | Purpose |
|-------------------|-----|---------|
| **Replicate Web UI** | http://localhost:8000 | Primary zine-style search & streaming RAG panel |
| **Interactive API Docs** | http://localhost:8000/docs | Swagger interactive REST endpoint exploration |
| **Gradio Console UI** | http://localhost:7861 | Simplified streaming chat workspace |
| **Langfuse Dashboard** | http://localhost:3000 | Tracing, telemetry, token logging, and user feedback |
| **Apache Airflow** | http://localhost:8080 | DAG pipelines orchestration and schedule monitoring |
| **OpenSearch Dashboards** | http://localhost:5601 | Search engine cluster analytics & index visualizations |

*Note: For Apache Airflow credentials, locate the generated username and password within `airflow/simple_auth_manager_passwords.json.generated`.*

---

## ⚙️ Development Guide

### 📂 Directory Layout
```
arxiv-paper-curator/
├── src/                    # Primary application package
│   ├── db/                 # Database initialization and factories
│   ├── models/             # SQLAlchemy ORM declarations
│   ├── repositories/       # Database query abstractions
│   ├── schemas/            # Pydantic validation schemas
│   ├── routers/            # FastAPI REST endpoints
│   └── services/           # Service connectors (Ollama, OpenSearch, Agents, Cache)
├── static/                 # Static web files (index.html, index.css, index.js)
├── airflow/                # Airflow workflow DAG configurations
├── tests/                  # Pytest automated testing suite
└── compose.yml             # Docker infrastructure orchestration
```

### 🔧 Commands
Use the pre-configured Makefile for development workflows:

```bash
make start       # Start all docker compose infrastructure services
make stop        # Stop compose containers and teardown networks
make restart     # Restart compose services
make logs        # Tail Docker compose container logs
make test        # Run the complete test suite (118 tests)
make test-cov    # Run test coverage analysis
make format      # Run Ruff formatter
make lint        # Run Ruff linter and MyPy typechecks
```

---

## 📡 API Endpoints Reference

| Endpoint | Method | Input Schema | Description |
|----------|--------|--------------|-------------|
| `/api/v1/health` | GET | - | Health status of system dependencies |
| `/api/v1/papers` | GET | - | List indexed arXiv papers from database |
| `/api/v1/hybrid-search/` | POST | `HybridSearchRequest` | Query OpenSearch using BM25, Vector, or fused RRF |
| `/api/v1/ask` | POST | `AskRequest` | Core RAG question-answering |
| `/api/v1/stream` | POST | `AskRequest` | Streaming SSE tokens RAG question-answering |
| `/api/v1/ask-agentic` | POST | `AskRequest` | Multi-agent adaptive retrieval RAG workflow |
| `/api/v1/feedback` | POST | `FeedbackRequest` | Submit trace rating score to Langfuse dashboard |

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
