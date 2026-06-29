import time
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.antigravity_rag.config_parser import get_config

_embeddings_model = None
_reranker_model = None

def get_embeddings_model() -> SentenceTransformer:
    global _embeddings_model
    if _embeddings_model is None:
        config = get_config()
        model_name = config.processing.get("embedding_model", "BAAI/bge-small-en-v1.5")
        print(f"Loading local SentenceTransformer model: {model_name}...")
        start_time = time.time()
        # This will download the model to HF cache on first run
        _embeddings_model = SentenceTransformer(model_name)
        print(f"Loaded embedding model in {time.time() - start_time:.2f}s")
    return _embeddings_model

def get_reranker_model() -> CrossEncoder:
    global _reranker_model
    if _reranker_model is None:
        print("Loading local CrossEncoder model: cross-encoder/ms-marco-MiniLM-L-6-v2...")
        start_time = time.time()
        _reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print(f"Loaded reranker model in {time.time() - start_time:.2f}s")
    return _reranker_model

def embed_text(texts: List[str]) -> List[List[float]]:
    model = get_embeddings_model()
    # Batch size is 32
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)
    # Convert numpy arrays to lists
    return [x.tolist() for x in embeddings]

def rerank(query: str, chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    if not chunks:
        return []
        
    model = get_reranker_model()
    
    # Form pairs: [query, doc_text]
    pairs = [[query, chunk["chunk_text"]] for chunk in chunks]
    
    # Predict relevance scores
    scores = model.predict(pairs)
    
    # Assign score to each chunk
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)
        
    # Sort chunks by rerank score descending
    sorted_chunks = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    
    # Return top_k
    return sorted_chunks[:top_k]
