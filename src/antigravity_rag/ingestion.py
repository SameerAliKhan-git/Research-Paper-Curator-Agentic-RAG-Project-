import os
import re
import uuid
import httpx
import fitz  # PyMuPDF
import urllib.request
from datetime import datetime, timedelta
import concurrent.futures
import difflib
from typing import List, Dict, Any, Optional, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.antigravity_rag.config_parser import get_config
from src.antigravity_rag.db_sqlite import insert_paper, insert_chunks, update_paper_indexed_status, paper_exists, doi_exists
from src.antigravity_rag.db_qdrant import upsert_chunks
from src.antigravity_rag.local_embeddings import embed_text
import sys

def safe_print(*args, **kwargs):
    sep = kwargs.get('sep', ' ')
    end = kwargs.get('end', '\n')
    file = kwargs.get('file', sys.stdout)
    
    msg = sep.join(str(arg) for arg in args)
    try:
        file.write(msg + end)
        file.flush()
    except UnicodeEncodeError:
        try:
            encoding = getattr(file, 'encoding', None) or 'utf-8'
            safe_msg = msg.encode(encoding, errors='replace').decode(encoding)
            file.write(safe_msg + end)
            file.flush()
        except Exception:
            try:
                ascii_msg = msg.encode('ascii', errors='ignore').decode('ascii')
                file.write(ascii_msg + end)
                file.flush()
            except Exception:
                pass
    except Exception:
        pass

print = safe_print

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

def refine_query(query: str) -> str:
    clean = query.lower()
    for prefix in ["what is a ", "what is ", "what are ", "how does ", "explain ", "tell me about ", "show me "]:
        clean = clean.replace(prefix, "")
    clean = re.sub(r'[\?\.!]', '', clean).strip()
    if "vector db" in clean:
        clean = clean.replace("vector db", "vector database")
    return clean

def search_arxiv(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    results = []
    if not ARXIV_AVAILABLE:
        print("arxiv library not installed. Skipping ArXiv search.")
        return results
        
    try:
        refined = refine_query(query)
        # Restrict to CS domains to prevent dark matter/physics search contamination
        arxiv_query = f'("{refined}" OR "{refined.replace("database", "db")}") AND (cat:cs.AI OR cat:cs.LG OR cat:cs.DB OR cat:cs.IR OR cat:cs.CL)'
        
        client = arxiv.Client()
        search = arxiv.Search(
            query=arxiv_query,
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
    refined = refine_query(query)
    params = {
        "query": f'"{refined}" database',
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
    
    clean_q = query.lower()
    # Simple classifier: check if queries relate to patent/author profiles (Google Scholar domain)
    # vs. scientific computer science papers (ArXiv / Semantic Scholar domain)
    query_domain = "science"
    if any(keyword in clean_q for keyword in ["patent", "author:", "profile", "citations of", "h-index"]):
        query_domain = "scholar"
        
    tasks = []
    # Using thread pool to run parallel queries dynamically
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        if query_domain == "scholar":
            print(f"Ingestion classifier: Routing query '{query}' to Google Scholar")
            if sources_cfg.get("google_scholar", {}).get("enabled", True):
                tasks.append(executor.submit(search_google_scholar, query, 3))
        else:
            print(f"Ingestion classifier: Routing query '{query}' to ArXiv and Semantic Scholar")
            if sources_cfg.get("arxiv", {}).get("enabled", True):
                tasks.append(executor.submit(search_arxiv, query, limit))
            if sources_cfg.get("semantic_scholar", {}).get("enabled", True):
                tasks.append(executor.submit(search_semantic_scholar, query, limit))
            
        concurrent.futures.wait(tasks)
        
    all_results = []
    for t in tasks:
        all_results.extend(t.result())
        
    # Deduplicate results by title similarity
    deduplicated = []
    for paper in all_results:
        is_dup = False
        for existing in deduplicated:
            ratio = difflib.SequenceMatcher(None, paper["title"].lower(), existing["title"].lower()).ratio()
            if ratio > 0.9:
                is_dup = True
                break
        if not is_dup:
            deduplicated.append(paper)
            
    return deduplicated[:5]

def fetch_pdf(paper: Dict[str, Any]) -> Optional[bytes]:
    """Downloads PDF to memory bytes if pdf_url is present."""
    pdf_url = paper.get("pdf_url")
    if not pdf_url:
        return None
        
    try:
        print(f"Downloading PDF: {pdf_url}")
        # Standard requests get with 5s timeout
        # Add headers to act as a browser
        headers = {"User-Agent": "Mozilla/5.0"}
        
        with httpx.Client(follow_redirects=True) as client:
            resp = client.get(pdf_url, headers=headers, timeout=5.0)
            if resp.status_code == 200:
                return resp.content
            else:
                print(f"Failed to download PDF. Status code: {resp.status_code}")
    except Exception as e:
        print(f"Error downloading PDF: {e}")
        
    return None

def extract_text_from_pdf(pdf_bytes: bytes) -> List[Tuple[int, str]]:
    """Extract and clean text page-by-page from PDF bytes using PyMuPDF."""
    pages_text = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            # Clean text
            # Remove headers/footers (simple numeric footers, etc.)
            cleaned = re.sub(r'\n\d+\s*\n', '\n', text)
            # Normalize spaces
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            
            pages_text.append((page_num + 1, cleaned))
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
    return pages_text

def process_and_index_paper(paper: Dict[str, Any]) -> bool:
    """Ingests paper metadata, chunks full text (or abstract), embeds, and indexes it."""
    # Check duplicate in database
    if paper_exists(paper["paper_id"]) or (paper.get("doi") and doi_exists(paper["doi"])):
        print(f"Paper '{paper['title']}' already indexed. Skipping.")
        return True
        
    # 1. Fetch PDF
    pdf_bytes = fetch_pdf(paper)
    
    # 2. Save paper metadata in SQLite with PDF BLOB
    inserted = insert_paper(
        paper_id=paper["paper_id"],
        title=paper["title"],
        authors=paper["authors"],
        year=paper["year"],
        source=paper["source"],
        url=paper["url"],
        doi=paper.get("doi"),
        abstract=paper["abstract"],
        full_text_path=None,
        pdf_blob=pdf_bytes
    )
    
    # 3. Extract text page-by-page
    pages = []
    if pdf_bytes:
        pages = extract_text_from_pdf(pdf_bytes)
        
    # 4. Chunk text
    config = get_config()
    chunk_size = config.processing.get("chunk_size", 512)
    chunk_overlap = config.processing.get("chunk_overlap", 50)
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks_to_insert = []
    
    if pages:
        global_chunk_idx = 0
        for page_number, page_text in pages:
            if not page_text:
                continue
            page_chunks = splitter.split_text(page_text)
            current_pos = 0
            for chunk_text in page_chunks:
                chunk_id = f"{paper['paper_id']}_{global_chunk_idx}"
                
                # Calculate character offsets
                start_char = page_text.find(chunk_text, current_pos)
                if start_char == -1:
                    start_char = page_text.find(chunk_text)
                    
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
                    "chunk_index": global_chunk_idx,
                    "chunk_text": chunk_text,
                    "start_char": start_char,
                    "end_char": end_char,
                    "section_title": f"Page {page_number}",
                    "token_count": token_count,
                    "page_number": page_number
                })
                global_chunk_idx += 1
    else:
        # Fallback to abstract (Page 1)
        abstract_text = paper.get("abstract", "")
        if abstract_text:
            raw_chunks = splitter.split_text(abstract_text)
            for idx, chunk_text in enumerate(raw_chunks):
                chunk_id = f"{paper['paper_id']}_{idx}"
                token_count = len(chunk_text.split())
                chunks_to_insert.append({
                    "chunk_id": chunk_id,
                    "paper_id": paper["paper_id"],
                    "chunk_index": idx,
                    "chunk_text": chunk_text,
                    "start_char": 0,
                    "end_char": len(chunk_text),
                    "section_title": "Abstract",
                    "token_count": token_count,
                    "page_number": 1
                })
                
    if not chunks_to_insert:
        print(f"No text content found to index for {paper['title']}. Skipping.")
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
