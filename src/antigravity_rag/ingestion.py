import os
import re
import uuid
import httpx
import fitz  # PyMuPDF
import urllib.request
from datetime import datetime, timedelta
import concurrent.futures
import difflib
from typing import List, Dict, Any, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.antigravity_rag.config_parser import get_config
from src.antigravity_rag.db_sqlite import insert_paper, insert_chunks, update_paper_indexed_status, paper_exists, doi_exists
from src.antigravity_rag.db_qdrant import upsert_chunks
from src.antigravity_rag.local_embeddings import embed_text

# Safe Scholarly imports
try:
    from scholarly import scholarly
    SCHOLARLY_AVAILABLE = True
except ImportError:
    SCHOLARLY_AVAILABLE = False

# Safe Arxiv imports
try:
    import arxiv
    ARXIV_AVAILABLE = True
except ImportError:
    ARXIV_AVAILABLE = False

def search_arxiv(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    results = []
    if not ARXIV_AVAILABLE:
        print("arxiv library not installed. Skipping ArXiv search.")
        return results
        
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        for r in client.results(search):
            paper_id = f"arxiv_{r.get_short_id()}"
            authors = ", ".join([a.name for a in r.authors])
            year = r.published.year if r.published else None
            
            results.append({
                "paper_id": paper_id,
                "title": r.title,
                "authors": authors,
                "year": year,
                "source": "arxiv",
                "url": r.entry_id,
                "doi": r.doi,
                "abstract": r.summary,
                "pdf_url": r.pdf_url
            })
    except Exception as e:
        print(f"Error searching ArXiv: {e}")
        
    return results

def search_semantic_scholar(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    results = []
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": max_results,
        "fields": "title,authors,year,url,externalIds,abstract,openAccessPdf"
    }
    try:
        resp = httpx.get(url, params=params, timeout=10.0)
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            for r in data:
                external_ids = r.get("externalIds", {})
                doi = external_ids.get("DOI")
                arxiv_id = external_ids.get("ArXiv")
                
                paper_id_val = arxiv_id if arxiv_id else r.get("paperId")
                paper_id = f"sem_{paper_id_val}"
                
                authors = ", ".join([a.get("name", "") for a in r.get("authors", [])])
                
                pdf_url = None
                oa_pdf = r.get("openAccessPdf")
                if oa_pdf and isinstance(oa_pdf, dict):
                    pdf_url = oa_pdf.get("url")
                
                results.append({
                    "paper_id": paper_id,
                    "title": r.get("title", "Untitled"),
                    "authors": authors,
                    "year": r.get("year"),
                    "source": "semantic_scholar",
                    "url": r.get("url"),
                    "doi": doi,
                    "abstract": r.get("abstract", ""),
                    "pdf_url": pdf_url
                })
    except Exception as e:
        print(f"Error searching Semantic Scholar: {e}")
    return results

def search_google_scholar(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    results = []
    if not SCHOLARLY_AVAILABLE:
        print("scholarly library not installed. Skipping Google Scholar.")
        return results
        
    try:
        search_query = scholarly.search_pubs(query)
        for _ in range(max_results):
            try:
                pub = next(search_query)
                bib = pub.get("bib", {})
                title = bib.get("title", "Untitled")
                authors = ", ".join(bib.get("author", []))
                year = int(bib.get("pub_year")) if bib.get("pub_year") else None
                url = pub.get("pub_url")
                
                paper_id = f"gs_{uuid.uuid5(uuid.NAMESPACE_DNS, title)}"
                
                results.append({
                    "paper_id": paper_id,
                    "title": title,
                    "authors": authors,
                    "year": year,
                    "source": "google_scholar",
                    "url": url,
                    "doi": None,
                    "abstract": bib.get("abstract", ""),
                    "pdf_url": pub.get("eprint_url")  # eprint_url is often a direct PDF link
                })
            except StopIteration:
                break
    except Exception as e:
        print(f"Error searching Google Scholar: {e}")
    return results

def search_all_sources(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    config = get_config()
    sources_cfg = config.sources
    
    tasks = []
    # Using thread pool to run parallel queries
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        if sources_cfg.get("arxiv", {}).get("enabled", True):
            tasks.append(executor.submit(search_arxiv, query, limit))
        if sources_cfg.get("semantic_scholar", {}).get("enabled", True):
            tasks.append(executor.submit(search_semantic_scholar, query, limit))
        if sources_cfg.get("google_scholar", {}).get("enabled", False):
            tasks.append(executor.submit(search_google_scholar, query, 2))
            
        concurrent.futures.wait(tasks)
        
    all_results = []
    for t in tasks:
        all_results.extend(t.result())
        
    # Deduplicate results by title similarity
    deduplicated = []
    for paper in all_results:
        # Check if already in deduplicated
        is_dup = False
        for existing in deduplicated:
            ratio = difflib.SequenceMatcher(None, paper["title"].lower(), existing["title"].lower()).ratio()
            if ratio > 0.9:
                is_dup = True
                break
        if not is_dup:
            deduplicated.append(paper)
            
    # Sort and return top 5
    return deduplicated[:5]

def fetch_pdf(paper: Dict[str, Any]) -> Optional[str]:
    """Downloads PDF to local papers_store if pdf_url is present."""
    pdf_url = paper.get("pdf_url")
    if not pdf_url:
        return None
        
    config = get_config()
    store_dir = config.storage.get("papers_store", "./papers_store")
    
    source_dir = os.path.join(store_dir, paper["source"])
    os.makedirs(source_dir, exist_ok=True)
    
    # Sanitize paper_id for filename
    safe_id = re.sub(r'[^a-zA-Z0-9_\-]', '_', paper["paper_id"])
    file_path = os.path.join(source_dir, f"{safe_id}.pdf")
    
    # Don't re-download if it already exists
    if os.path.exists(file_path):
        return file_path
        
    try:
        print(f"Downloading PDF: {pdf_url} -> {file_path}")
        # Standard requests get with 5s timeout
        # Add headers to act as a browser
        headers = {"User-Agent": "Mozilla/5.0"}
        
        with httpx.Client(follow_redirects=True) as client:
            resp = client.get(pdf_url, headers=headers, timeout=5.0)
            if resp.status_code == 200:
                with open(file_path, "wb") as f:
                    f.write(resp.content)
                return file_path
            else:
                print(f"Failed to download PDF. Status code: {resp.status_code}")
    except Exception as e:
        print(f"Error downloading PDF: {e}")
        
    return None

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract and clean text from PDF using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        text_pages = []
        for page in doc:
            text_pages.append(page.get_text())
            
        full_text = "\n".join(text_pages)
        
        # Clean text
        # Remove headers/footers (simple numeric footers, etc.)
        cleaned = re.sub(r'\n\d+\s*\n', '\n', full_text)
        # Normalize spaces
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return ""

def process_and_index_paper(paper: Dict[str, Any]) -> bool:
    """Ingests paper metadata, chunks full text (or abstract), embeds, and indexes it."""
    # Check duplicate in database
    if paper_exists(paper["paper_id"]) or (paper.get("doi") and doi_exists(paper["doi"])):
        print(f"Paper '{paper['title']}' already indexed. Skipping.")
        return True
        
    # 1. Fetch PDF
    pdf_path = fetch_pdf(paper)
    paper["full_text_path"] = pdf_path
    
    # 2. Save paper metadata in SQLite
    inserted = insert_paper(
        paper_id=paper["paper_id"],
        title=paper["title"],
        authors=paper["authors"],
        year=paper["year"],
        source=paper["source"],
        url=paper["url"],
        doi=paper.get("doi"),
        abstract=paper["abstract"],
        full_text_path=pdf_path
    )
    
    # 3. Extract full text or fallback to abstract
    text_to_index = ""
    if pdf_path:
        text_to_index = extract_text_from_pdf(pdf_path)
        
    if not text_to_index:
        print(f"Using abstract as index text for {paper['title']}")
        text_to_index = paper["abstract"]
        
    if not text_to_index:
        print(f"No text content found to index for {paper['title']}. Skipping chunking.")
        return False
        
    # 4. Chunk text
    config = get_config()
    chunk_size = config.processing.get("chunk_size", 512)
    chunk_overlap = config.processing.get("chunk_overlap", 50)
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    raw_chunks = splitter.split_text(text_to_index)
    
    chunks_to_insert = []
    current_pos = 0
    
    for idx, chunk_text in enumerate(raw_chunks):
        chunk_id = f"{paper['paper_id']}_{idx}"
        
        # Calculate character offsets
        start_char = text_to_index.find(chunk_text, current_pos)
        if start_char == -1:
            start_char = text_to_index.find(chunk_text)
            
        if start_char != -1:
            end_char = start_char + len(chunk_text)
            current_pos = end_char
        else:
            start_char = 0
            end_char = len(chunk_text)
            
        # Count words as token approximation
        token_count = len(chunk_text.split())
        
        chunks_to_insert.append({
            "chunk_id": chunk_id,
            "paper_id": paper["paper_id"],
            "chunk_index": idx,
            "chunk_text": chunk_text,
            "start_char": start_char,
            "end_char": end_char,
            "section_title": None,
            "token_count": token_count
        })
        
    if not chunks_to_insert:
        return False
        
    # 5. Embed chunks
    chunk_texts = [c["chunk_text"] for c in chunks_to_insert]
    try:
        embeddings = embed_text(chunk_texts)
    except Exception as e:
        print(f"Error embedding chunks: {e}")
        return False
        
    # 6. Index in SQLite & Qdrant
    try:
        # SQLite
        insert_chunks(chunks_to_insert)
        # Qdrant
        upsert_chunks(chunks_to_insert, embeddings)
        
        # Mark as indexed
        update_paper_indexed_status(paper["paper_id"], True)
        print(f"Successfully indexed paper: {paper['title']}")
        return True
    except Exception as e:
        print(f"Error indexing chunks in storage: {e}")
        return False
