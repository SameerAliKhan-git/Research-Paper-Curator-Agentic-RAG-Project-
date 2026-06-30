import sqlite3
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from src.antigravity_rag.config_parser import get_config

def get_db_connection():
    config = get_config()
    db_path = config.storage.get("sqlite_db", "./papers.db")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    # Check if pdf_blob column exists in papers table
    config = get_config()
    db_path = config.storage.get("sqlite_db", "./papers.db")
    
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT pdf_blob FROM papers LIMIT 1")
            conn.close()
        except sqlite3.OperationalError:
            conn.close()
            print("Outdated database schema detected. Rebuilding papers.db...")
            try:
                os.remove(db_path)
            except Exception as ex:
                print(f"Failed to delete old DB: {ex}")
                
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create papers table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS papers (
        paper_id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        authors TEXT,
        year INTEGER,
        source TEXT,
        url TEXT,
        doi TEXT UNIQUE,
        abstract TEXT,
        full_text_path TEXT,
        pdf_blob BLOB,
        ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        indexed BOOLEAN DEFAULT 0
    );
    """)
    
    # Create chunks table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id TEXT PRIMARY KEY,
        paper_id TEXT NOT NULL REFERENCES papers(paper_id),
        chunk_index INTEGER,
        chunk_text TEXT NOT NULL,
        start_char INTEGER,
        end_char INTEGER,
        section_title TEXT,
        token_count INTEGER,
        page_number INTEGER
    );
    """)
    
    # Create FTS5 virtual table
    try:
        cursor.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(chunk_id, chunk_text);")
    except sqlite3.OperationalError as e:
        print(f"Failed to create FTS5 table, attempting FTS4: {e}")
        cursor.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts4(chunk_id, chunk_text);")

    # Create query_cache table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS query_cache (
        query TEXT PRIMARY KEY,
        results_json TEXT NOT NULL,
        cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    conn.commit()
    conn.close()

def insert_paper(
    paper_id: str,
    title: str,
    authors: str,
    year: int,
    source: str,
    url: str,
    doi: Optional[str],
    abstract: str,
    full_text_path: Optional[str] = None,
    pdf_blob: Optional[bytes] = None
) -> bool:
    import zlib
    if pdf_blob:
        try:
            pdf_blob = zlib.compress(pdf_blob)
        except Exception as e:
            print(f"Warning: Failed to compress PDF blob: {e}")
            
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
        INSERT OR IGNORE INTO papers (paper_id, title, authors, year, source, url, doi, abstract, full_text_path, pdf_blob)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (paper_id, title, authors, year, source, url, doi, abstract, full_text_path, pdf_blob))
        conn.commit()
        success = cursor.rowcount > 0
        return success
    except Exception as e:
        print(f"Error inserting paper: {e}")
        return False
    finally:
        conn.close()

def insert_chunks(chunks_list: List[Dict[str, Any]]):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        for chunk in chunks_list:
            cursor.execute("""
            INSERT OR REPLACE INTO chunks (chunk_id, paper_id, chunk_index, chunk_text, start_char, end_char, section_title, token_count, page_number)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                chunk["chunk_id"],
                chunk["paper_id"],
                chunk["chunk_index"],
                chunk["chunk_text"],
                chunk.get("start_char"),
                chunk.get("end_char"),
                chunk.get("section_title"),
                chunk.get("token_count"),
                chunk.get("page_number")
            ))
            
            cursor.execute("""
            INSERT OR REPLACE INTO chunks_fts (chunk_id, chunk_text)
            VALUES (?, ?)
            """, (chunk["chunk_id"], chunk["chunk_text"]))
            
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Error inserting chunks: {e}")
        raise e
    finally:
        conn.close()

def get_paper_pdf(paper_id: str) -> Optional[bytes]:
    import zlib
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT pdf_blob FROM papers WHERE paper_id = ?", (paper_id,))
        row = cursor.fetchone()
        if not row or not row["pdf_blob"]:
            return None
        blob = row["pdf_blob"]
        try:
            return zlib.decompress(blob)
        except zlib.error:
            # Fallback for old uncompressed database records
            return blob
    except Exception as e:
        print(f"Error getting PDF from DB: {e}")
        return None
    finally:
        conn.close()

def paper_exists(paper_id: str) -> bool:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM papers WHERE paper_id = ?", (paper_id,))
    res = cursor.fetchone()
    conn.close()
    return res is not None

def doi_exists(doi: str) -> bool:
    if not doi:
        return False
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM papers WHERE doi = ?", (doi,))
    res = cursor.fetchone()
    conn.close()
    return res is not None

def search_fts(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    import re
    clean_query = re.sub(r'[^\w\s]', ' ', query)
    clean_query = " ".join(clean_query.split())
    
    conn = get_db_connection()
    cursor = conn.cursor()
    results = []
    try:
        cursor.execute("""
        SELECT f.chunk_id, f.chunk_text, c.paper_id, c.chunk_index, c.start_char, c.end_char, c.section_title, c.page_number,
               p.title as paper_title, p.authors, p.year, p.url, p.full_text_path
        FROM chunks_fts f
        JOIN chunks c ON f.chunk_id = c.chunk_id
        JOIN papers p ON c.paper_id = p.paper_id
        WHERE chunks_fts MATCH ?
        LIMIT ?
        """, (clean_query, top_k))
        
        rows = cursor.fetchall()
        for row in rows:
            results.append(dict(row))
    except Exception as e:
        print(f"FTS search error (trying fallback LIKE): {e}")
        try:
            cursor.execute("""
            SELECT c.chunk_id, c.chunk_text, c.paper_id, c.chunk_index, c.start_char, c.end_char, c.section_title, c.page_number,
                   p.title as paper_title, p.authors, p.year, p.url, p.full_text_path
            FROM chunks c
            JOIN papers p ON c.paper_id = p.paper_id
            WHERE c.chunk_text LIKE ?
            LIMIT ?
            """, (f"%{clean_query}%", top_k))
            rows = cursor.fetchall()
            for row in rows:
                results.append(dict(row))
        except Exception as ex:
            print(f"LIKE search error: {ex}")
    finally:
        conn.close()
    return results

def get_paper_metadata(paper_id: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM papers WHERE paper_id = ?", (paper_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_chunk(chunk_id: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT c.*, p.title as paper_title, p.authors, p.year, p.url, p.full_text_path
    FROM chunks c
    JOIN papers p ON c.paper_id = p.paper_id
    WHERE c.chunk_id = ?
    """, (chunk_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_all_papers() -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM papers ORDER BY ingested_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_paper_chunks(paper_id: str) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM chunks WHERE paper_id = ? ORDER BY chunk_index ASC", (paper_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def update_paper_indexed_status(paper_id: str, status: bool):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE papers SET indexed = ? WHERE paper_id = ?", (1 if status else 0, paper_id))
    conn.commit()
    conn.close()

def get_cached_query_results(query: str) -> Optional[List[Dict[str, Any]]]:
    import json
    query = query.strip().lower()
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT results_json FROM query_cache WHERE query = ?", (query,))
        row = cursor.fetchone()
        if row:
            return json.loads(row["results_json"])
    except Exception as e:
        print(f"Error reading query cache: {e}")
    finally:
        conn.close()
    return None

def save_cached_query_results(query: str, results: List[Dict[str, Any]]):
    import json
    query = query.strip().lower()
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        results_json = json.dumps(results)
        cursor.execute("""
            INSERT OR REPLACE INTO query_cache (query, results_json, cached_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        """, (query, results_json))
        conn.commit()
    except Exception as e:
        print(f"Error saving query cache: {e}")
    finally:
        conn.close()
