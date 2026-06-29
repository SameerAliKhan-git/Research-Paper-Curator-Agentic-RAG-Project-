from typing import List, Dict, Any
from src.antigravity_rag.config_parser import get_config
from src.antigravity_rag.db_sqlite import get_db_connection, search_fts
from src.antigravity_rag.db_qdrant import search_vectors
from src.antigravity_rag.local_embeddings import embed_text, rerank

def hybrid_search(query: str) -> List[Dict[str, Any]]:
    config = get_config()
    retrieval_cfg = config.retrieval
    
    dense_top_k = retrieval_cfg.get("dense_top_k", 20)
    sparse_top_k = retrieval_cfg.get("sparse_top_k", 10)
    rerank_top_k = retrieval_cfg.get("rerank_top_k", 5)
    
    # 1. Sparse search (SQLite FTS)
    sparse_results = search_fts(query, top_k=sparse_top_k)
    
    # 2. Dense search (Qdrant)
    # Get query embedding
    query_vector = embed_text([query])[0]
    dense_results = search_vectors(query_vector, top_k=dense_top_k)
    
    # 3. Reciprocal Rank Fusion (RRF)
    # RRF params
    rrf_k = 60
    rrf_scores = {}
    
    # Track rank lists
    for rank, item in enumerate(dense_results):
        chunk_id = item["chunk_id"]
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (rrf_k + rank + 1))
        
    for rank, item in enumerate(sparse_results):
        chunk_id = item["chunk_id"]
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (rrf_k + rank + 1))
        
    # Sort chunk_ids by RRF score descending
    sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    
    # We take top (dense_top_k + sparse_top_k) candidates for reranking
    candidate_ids = sorted_chunk_ids[:(dense_top_k + sparse_top_k)]
    
    if not candidate_ids:
        return []
        
    # 4. Fetch full chunk details for candidates from SQLite
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Construct placeholders
    placeholders = ",".join("?" for _ in candidate_ids)
    query_str = f"""
    SELECT c.chunk_id, c.chunk_text, c.paper_id, c.chunk_index, c.start_char, c.end_char, c.section_title,
           p.title as paper_title, p.authors, p.year, p.url, p.full_text_path
    FROM chunks c
    JOIN papers p ON c.paper_id = p.paper_id
    WHERE c.chunk_id IN ({placeholders})
    """
    
    cursor.execute(query_str, candidate_ids)
    rows = cursor.fetchall()
    conn.close()
    
    # Map row by chunk_id
    chunk_map = {row["chunk_id"]: dict(row) for row in rows}
    
    # Build candidate chunks in RRF sorted order
    candidate_chunks = []
    for chunk_id in candidate_ids:
        if chunk_id in chunk_map:
            chunk_data = chunk_map[chunk_id]
            # Add RRF score
            chunk_data["rrf_score"] = rrf_scores[chunk_id]
            candidate_chunks.append(chunk_data)
            
    # 5. Local Cross-Encoder reranking
    reranked_chunks = rerank(query, candidate_chunks, top_k=rerank_top_k)
    
    return reranked_chunks
