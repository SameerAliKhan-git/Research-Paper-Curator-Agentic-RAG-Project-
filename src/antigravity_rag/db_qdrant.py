import os
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from src.antigravity_rag.config_parser import get_config

_qdrant_client = None

def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        config = get_config()
        qdrant_path = config.storage.get("qdrant_path", "./qdrant_storage")
        
        # Use path-based persistent storage for local-only lightweight execution
        _qdrant_client = QdrantClient(path=qdrant_path)
    return _qdrant_client

def init_qdrant():
    client = get_qdrant_client()
    collection_name = "research_papers"
    
    # Check if collection exists
    collections = client.get_collections().collections
    exists = any(c.name == collection_name for c in collections)
    
    if not exists:
        print(f"Creating Qdrant collection: {collection_name}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=384,  # size of BAAI/bge-small-en-v1.5 embeddings
                distance=Distance.COSINE
            )
        )
    else:
        print(f"Qdrant collection '{collection_name}' already exists.")

def upsert_chunks(chunks_list: List[Dict[str, Any]], embeddings_list: List[List[float]]):
    if len(chunks_list) != len(embeddings_list):
        raise ValueError("The number of chunks and embeddings must match.")
        
    client = get_qdrant_client()
    collection_name = "research_papers"
    
    points = []
    for idx, (chunk, embedding) in enumerate(zip(chunks_list, embeddings_list)):
        # Generate an integer point ID from chunk_id (since Qdrant IDs must be uuid or integer)
        # Using hash of chunk_id or simple integer representation
        # Wait, qdrant-client can accept string IDs if they are UUID format,
        # or we can use a hash / random uuid
        import uuid
        
        # Consistent UUID from chunk_id string
        uid = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["chunk_id"]))
        
        # Extract first 200 chars as preview
        text_preview = chunk["chunk_text"][:200]
        
        points.append(PointStruct(
            id=uid,
            vector=embedding,
            payload={
                "chunk_id": chunk["chunk_id"],
                "paper_id": chunk["paper_id"],
                "chunk_index": chunk["chunk_index"],
                "text_preview": text_preview
            }
        ))
        
    if points:
        client.upsert(
            collection_name=collection_name,
            points=points
        )
        print(f"Successfully upserted {len(points)} points to Qdrant.")

def search_vectors(query_vector: List[float], top_k: int = 20) -> List[Dict[str, Any]]:
    client = get_qdrant_client()
    collection_name = "research_papers"
    
    # Query collection using the modern query_points API
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k
    )
    
    results = []
    for hit in response.points:
        results.append({
            "chunk_id": hit.payload.get("chunk_id"),
            "paper_id": hit.payload.get("paper_id"),
            "chunk_index": hit.payload.get("chunk_index"),
            "score": hit.score
        })
        
    return results
