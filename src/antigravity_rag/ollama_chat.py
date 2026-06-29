import httpx
import re
import json
import logging
from typing import List, Dict, Any, Tuple
from src.antigravity_rag.config_parser import get_config
from src.antigravity_rag.db_sqlite import get_chunk

logger = logging.getLogger(__name__)

def _resolve_model(ollama_url: str, requested_model: str) -> str:
    """Check available models in Ollama and fallback if requested_model is not available."""
    try:
        resp = httpx.get(f"{ollama_url}/api/tags", timeout=5.0)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            if requested_model in models:
                return requested_model
            # Try case-insensitive matching or suffix matching
            for m in models:
                if requested_model.lower() in m.lower() or m.lower() in requested_model.lower():
                    return m
            # Fallback to the first available model if any
            if models:
                print(f"Model '{requested_model}' not found in Ollama. Falling back to '{models[0]}'.")
                return models[0]
    except Exception as e:
        print(f"Error checking Ollama models: {e}")
    # Return requested model as default fallback
    return requested_model

def call_ollama(prompt: str) -> str:
    config = get_config()
    llm_cfg = config.llm
    
    ollama_url = llm_cfg.get("ollama_url", "http://localhost:11434")
    model_name = llm_cfg.get("model_name", "mistral:7b-instruct-v0.3-q4_K_M")
    temperature = llm_cfg.get("temperature", 0.2)
    max_tokens = llm_cfg.get("max_tokens", 1024)
    
    # Resolve actual model name
    active_model = _resolve_model(ollama_url, model_name)
    
    payload = {
        "model": active_model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }
    
    try:
        resp = httpx.post(f"{ollama_url}/api/generate", json=payload, timeout=60.0)
        if resp.status_code == 200:
            return resp.json().get("response", "")
        else:
            return f"Ollama error: HTTP {resp.status_code}"
    except Exception as e:
        return f"Ollama connection error: {e}"

def generate_answer(query: str, retrieved_chunks: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    # Format excerpts
    excerpts_text = ""
    for chunk in retrieved_chunks:
        excerpts_text += f"[{chunk['chunk_id']}] Title: {chunk['paper_title']}\n{chunk['chunk_text']}\n\n"
        
    prompt = f"""You are a research assistant. Answer the user's question using ONLY the provided paper excerpts.
Cite each factual statement with the chunk ID in the format 【chunk_id】.
If the excerpts are insufficient, say "I don't have enough information."

User question: {query}
Excerpts:
{excerpts_text}
"""
    
    raw_response = call_ollama(prompt)
    
    # Post-processing: extract citation markers
    # Matches patterns like 【chunk_id】 or 【chunk_id_index】 or even [chunk_id]
    # We prioritize the standard 【([^】]+)】 marker
    markers = re.findall(r'【([^】]+)】', raw_response)
    
    # Fallback to square brackets if no standard markers are found
    if not markers:
        markers = re.findall(r'\[([^\]]+)\]', raw_response)
        
    citation_map = {}
    citation_counter = 1
    
    processed_response = raw_response
    
    # De-duplicate markers while preserving order
    seen_markers = []
    for m in markers:
        if m not in seen_markers:
            seen_markers.append(m)
            
    for marker in seen_markers:
        # Get chunk details from DB or from the retrieved chunks list
        chunk_details = None
        for chunk in retrieved_chunks:
            if chunk["chunk_id"] == marker or chunk["chunk_id"].endswith(marker):
                chunk_details = chunk
                break
                
        if not chunk_details:
            # Try loading from database
            chunk_details = get_chunk(marker)
            
        if chunk_details:
            cite_num = str(citation_counter)
            citation_map[cite_num] = {
                "chunk_id": chunk_details["chunk_id"],
                "paper_title": chunk_details["paper_title"],
                "authors": chunk_details.get("authors", "Unknown"),
                "year": chunk_details.get("year", ""),
                "url": chunk_details.get("url", ""),
                "full_text_path": chunk_details.get("full_text_path", ""),
                "excerpt": chunk_details["chunk_text"],
                "start": chunk_details.get("start_char", 0),
                "end": chunk_details.get("end_char", 0)
            }
            
            # Replace marker with superscript HTML link/badge
            # We replace both standard and brackets style if present
            # We escape regex special characters in marker
            escaped_marker = re.escape(marker)
            
            # Replace standard marker
            processed_response = re.sub(
                rf'【{escaped_marker}】',
                f'<sup class="citation-badge" onclick="showCitation(\'{cite_num}\')" title="{chunk_details["paper_title"]} (Click to view excerpt)">[{cite_num}]</sup>',
                processed_response
            )
            
            # Replace bracket marker
            processed_response = re.sub(
                rf'\[{escaped_marker}\]',
                f'<sup class="citation-badge" onclick="showCitation(\'{cite_num}\')" title="{chunk_details["paper_title"]} (Click to view excerpt)">[{cite_num}]</sup>',
                processed_response
            )
            
            citation_counter += 1
            
    return processed_response, citation_map
