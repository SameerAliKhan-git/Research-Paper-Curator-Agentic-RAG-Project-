import os
import random

from locust import HttpUser, between, task, LoadTestShape


API_QUERIES = [
    "What are transformers in machine learning?",
    "How do large language models handle context?",
    "Explain the attention mechanism in neural networks.",
    "What is retrieval augmented generation?",
    "How does fine-tuning differ from prompt engineering?",
    "What are the limitations of current embedding models?",
    "How do vector databases improve semantic search?",
    "What is the difference between BM25 and dense retrieval?",
    "How do chain of thought prompting techniques work?",
    "What are the best practices for chunking documents in RAG?",
    "Explain the concept of knowledge distillation in LLMs.",
    "What are hallucinations in language models and how to mitigate them?",
    "How does cross-encoder reranking improve search quality?",
    "What is the role of prompt templates in RAG systems?",
    "How do multi-modal models process images and text together?",
]

SEARCH_QUERIES = [
    "transformer architecture",
    "neural network attention",
    "retrieval augmented generation",
    "large language model",
    "embedding similarity search",
    "knowledge graph integration",
    "text classification deep learning",
    "reinforcement learning from human feedback",
    "diffusion model generation",
    "graph neural network reasoning",
]

CATEGORIES = ["cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.NE", "stat.ML"]


class RAGUser(HttpUser):
    """Locust user simulating realistic traffic against the RAG API."""

    wait_time = between(1, 3)
    weight = 1

    def on_start(self):
        api_key = os.environ.get("LOCUST_API_KEY", "")
        if api_key:
            self.client.headers["X-API-Key"] = api_key

    @task(5)
    def ask_question(self):
        payload = {
            "query": random.choice(API_QUERIES),
            "top_k": random.randint(1, 5),
            "use_hybrid": random.choice([True, False]),
            "model": "llama3.2:1b",
        }
        if random.random() < 0.3:
            payload["categories"] = random.sample(CATEGORIES, k=random.randint(1, 2))
        self.client.post("/api/v1/ask", json=payload, name="/api/v1/ask")

    @task(3)
    def hybrid_search(self):
        payload = {
            "query": random.choice(SEARCH_QUERIES),
            "size": random.randint(5, 20),
            "from": 0,
            "use_hybrid": random.choice([True, False]),
            "latest_papers": random.choice([True, False]),
            "min_score": random.choice([0.0, 0.1, 0.3]),
        }
        if random.random() < 0.4:
            payload["categories"] = random.sample(CATEGORIES, k=random.randint(1, 2))
        self.client.post("/api/v1/hybrid-search/", json=payload, name="/api/v1/hybrid-search/")

    @task(2)
    def list_papers(self):
        limit = random.choice([10, 20, 50])
        offset = random.randint(0, 4) * limit
        self.client.get(f"/api/v1/papers/?limit={limit}&offset={offset}", name="/api/v1/papers/")

    @task(1)
    def paper_stats(self):
        self.client.get("/api/v1/papers/stats", name="/api/v1/papers/stats")

    @task(1)
    def health_check(self):
        self.client.get("/api/v1/health", name="/api/v1/health")

    @task(2)
    def agentic_ask(self):
        payload = {
            "query": random.choice(API_QUERIES),
            "model": "llama3.2:1b",
        }
        self.client.post("/api/v1/agentic-ask", json=payload, name="/api/v1/agentic-ask")

    @task(1)
    def related_papers(self):
        payload = {
            "query": random.choice(API_QUERIES),
            "top_k": random.randint(3, 10),
        }
        self.client.post("/api/v1/related", json=payload, name="/api/v1/related")


class FastLoadTestShape(LoadTestShape):
    """Staged load test shape with warmup, sustained, peak, and cooldown phases."""

    stages = [
        {"duration": 60, "users": 10, "spawn_rate": 2},
        {"duration": 180, "users": 50, "spawn_rate": 5},
        {"duration": 300, "users": 100, "spawn_rate": 10},
        {"duration": 360, "users": 20, "spawn_rate": 2},
    ]

    def tick(self):
        run_time = self.get_run_time()
        for stage in self.stages:
            if run_time < stage["duration"]:
                return stage["users"], stage["spawn_rate"]
        return None
