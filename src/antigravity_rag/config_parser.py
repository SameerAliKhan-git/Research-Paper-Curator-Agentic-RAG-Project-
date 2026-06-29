import os
from pathlib import Path
import yaml

class AntigravityConfig:
    def __init__(self, config_path: str = "config.yaml"):
        possible_paths = [
            Path(config_path),
            Path(__file__).parent.parent.parent / config_path,
            Path.cwd() / config_path
        ]
        
        self.config_data = {}
        loaded = False
        for path in possible_paths:
            if path.exists():
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        self.config_data = yaml.safe_load(f)
                    loaded = True
                    break
                except Exception as e:
                    print(f"Error loading config from {path}: {e}")
        
        if not loaded:
            self.config_data = {
                "sources": {
                    "arxiv": {"enabled": True, "categories": ["cs.AI", "cs.CL", "cs.LG"], "max_results": 10},
                    "semantic_scholar": {"enabled": True, "max_results": 5},
                    "google_scholar": {"enabled": False}
                },
                "schedule": {
                    "daily_ingestion_hour": 3,
                    "timezone": "UTC"
                },
                "processing": {
                    "chunk_size": 512,
                    "chunk_overlap": 50,
                    "embedding_model": "BAAI/bge-small-en-v1.5"
                },
                "llm": {
                    "model_name": "mistral:7b-instruct-v0.3-q4_K_M",
                    "ollama_url": "http://localhost:11434",
                    "temperature": 0.2,
                    "max_tokens": 1024
                },
                "retrieval": {
                    "dense_top_k": 20,
                    "sparse_top_k": 10,
                    "rerank_top_k": 5
                },
                "storage": {
                    "qdrant_path": "./qdrant_storage",
                    "papers_store": "./papers_store",
                    "sqlite_db": "./papers.db"
                }
            }

        self.sources = self.config_data.get("sources", {})
        self.schedule = self.config_data.get("schedule", {})
        self.processing = self.config_data.get("processing", {})
        self.llm = self.config_data.get("llm", {})
        self.retrieval = self.config_data.get("retrieval", {})
        self.storage = self.config_data.get("storage", {})

        os.makedirs(self.storage.get("qdrant_path", "./qdrant_storage"), exist_ok=True)
        os.makedirs(self.storage.get("papers_store", "./papers_store"), exist_ok=True)

_config_instance = None

def get_config() -> AntigravityConfig:
    global _config_instance
    if _config_instance is None:
        _config_instance = AntigravityConfig()
    return _config_instance
