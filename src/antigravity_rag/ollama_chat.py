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

def call_ollama(prompt: str, model_name: str = None) -> str:
    config = get_config()
    llm_cfg = config.llm
    
    ollama_url = llm_cfg.get("ollama_url", "http://localhost:11434")
    if not model_name:
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
        resp = httpx.post(f"{ollama_url}/api/generate", json=payload, timeout=180.0)
        if resp.status_code == 200:
            return resp.json().get("response", "")
        else:
            return f"Ollama error: HTTP {resp.status_code}"
    except Exception as e:
        return f"Ollama connection error: {e}"

def to_superscript(num_str: str) -> str:
    sup_map = {
        '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
        '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹'
    }
    return "".join(sup_map.get(c, c) for c in num_str)

def generate_answer(query: str, retrieved_chunks: List[Dict[str, Any]], model_name: str = None) -> Tuple[str, str, Dict[str, Any]]:
    # Format excerpts
    excerpts_text = ""
    for chunk in retrieved_chunks:
        excerpts_text += f"[{chunk['chunk_id']}] Title: {chunk['paper_title']}\n{chunk['chunk_text']}\n\n"
        
    prompt = f"""You are a research assistant. Answer the user's question using ONLY the provided paper excerpts.
If the excerpts are insufficient, say "I don't have enough information."

User question: {query}

Excerpts:
{excerpts_text}

Format your response EXACTLY as follows:
Thinking:
[Your step-by-step reasoning process, analyzing which chunks you are using and why]

Answer:
[Your final answer output text, citing each factual statement with the chunk ID in the format 【chunk_id】]

Thinking:
"""
    
    # We prepend 'Thinking:\n' back since we pre-fill it in the prompt to force the model to reason first
    raw_response = "Thinking:\n" + call_ollama(prompt, model_name=model_name)
    
    # Extract thinking and answer sections using a robust regex pattern
    thinking = ""
    cleaned_response = raw_response
    
    # Define a robust pattern matching standard and colloquial transitional headers
    answer_pat = r'\b(?:answer|final answer|response|based on the.*answer|output|answer text):\s*'
    
    if "thinking:" in raw_response.lower() and re.search(answer_pat, raw_response, re.IGNORECASE):
        # Split by Answer pattern
        parts = re.split(answer_pat, raw_response, flags=re.IGNORECASE)
        if len(parts) >= 2:
            thinking_part = parts[0]
            answer_part = "".join(parts[1:])
            
            # Extract thinking content
            match_think = re.search(r'\bthinking:\s*(.*)', thinking_part, re.DOTALL | re.IGNORECASE)
            if match_think:
                thinking = match_think.group(1).strip()
            else:
                thinking = thinking_part.replace("Thinking:", "").replace("thinking:", "").strip()
                
            cleaned_response = answer_part.strip()
    elif "thinking:" in raw_response.lower():
        # Only Thinking: was output, fallback
        parts = re.split(r'\bthinking:\s*', raw_response, flags=re.IGNORECASE)
        if len(parts) >= 2:
            thinking = parts[1].strip()
            cleaned_response = parts[1].strip()
            
    # Post-processing: extract citation markers
    markers = re.findall(r'【([^】]+)】', cleaned_response)
    if not markers:
        markers = re.findall(r'\[([^\]]+)\]', cleaned_response)
        
    citation_map = {}
    citation_counter = 1
    processed_response = cleaned_response
    
    seen_markers = []
    for m in markers:
        if m not in seen_markers:
            seen_markers.append(m)
            
    for marker in seen_markers:
        chunk_details = None
        for chunk in retrieved_chunks:
            if chunk["chunk_id"] == marker or chunk["chunk_id"].endswith(marker):
                chunk_details = chunk
                break
                
        if not chunk_details:
            chunk_details = get_chunk(marker)
            
        if chunk_details:
            cite_num = str(citation_counter)
            page_num = chunk_details.get("page_number", 1)
            paper_id = chunk_details.get("paper_id", "")
            
            citation_map[cite_num] = {
                "chunk_id": chunk_details["chunk_id"],
                "paper_id": paper_id,
                "paper_title": chunk_details["paper_title"],
                "authors": chunk_details.get("authors", "Unknown"),
                "year": chunk_details.get("year", ""),
                "url": chunk_details.get("url", ""),
                "page": page_num,
                "full_text_path": chunk_details.get("full_text_path", ""),
                "excerpt": chunk_details["chunk_text"],
                "start": chunk_details.get("start_char", 0),
                "end": chunk_details.get("end_char", 0)
            }
            
            cite_sup = to_superscript(cite_num)
            escaped_marker = re.escape(marker)
            
            pdf_url = f"http://localhost:8502/pdf/{paper_id}?page={page_num}"
            badge_html = f'<sup><a href="{pdf_url}" target="_blank" onclick="window.showCitation(\'{cite_num}\'); event.stopPropagation();" class="citation-badge" title="Open PDF: {chunk_details["paper_title"]} (Page {page_num})">{cite_num}📄</a></sup>'
            
            processed_response = re.sub(rf'【{escaped_marker}】', badge_html, processed_response)
            processed_response = re.sub(rf'\[{escaped_marker}\]', badge_html, processed_response)
            
            citation_counter += 1
            
    return processed_response, thinking, citation_map
