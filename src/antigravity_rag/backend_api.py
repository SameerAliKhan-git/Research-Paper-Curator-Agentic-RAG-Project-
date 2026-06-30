import os
import sys
import json
from fastapi import FastAPI, Response, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# Set project root path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.antigravity_rag.config_parser import get_config
from src.antigravity_rag.db_sqlite import get_paper_pdf, get_db_connection, init_db
from src.antigravity_rag.db_qdrant import get_qdrant_client
from src.antigravity_rag.ingestion import process_and_index_paper
from src.antigravity_rag.supervisor import stream_query_rag

# Initialize Database Schema
try:
    init_db()
except Exception as e:
    print(f"Warning: Database initialization failed: {e}")

app = FastAPI(title="Antigravity Unified Backend API")

# Enable CORS for React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/pdf/{paper_id}")
def serve_pdf(paper_id: str):
    pdf_bytes = get_paper_pdf(paper_id)
    if not pdf_bytes:
        raise HTTPException(status_code=404, detail="PDF not found")
    
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"inline; filename={paper_id}.pdf",
            "Access-Control-Allow-Origin": "*",
            "Content-Security-Policy": "frame-ancestors *"
        }
    )

@app.get("/api/papers")
def get_papers():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT paper_id, title, authors, year, url, full_text_path FROM papers ORDER BY rowid DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

@app.get("/api/chunks/{paper_id}")
def get_chunks(paper_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT chunk_id, chunk_text, chunk_index, page_number, section_title 
        FROM chunks 
        WHERE paper_id = ? 
        ORDER BY chunk_index ASC
    """, (paper_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

@app.get("/api/models")
def get_models():
    try:
        import httpx
        resp = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            return {"models": models}
        return {"models": ["llama3.2:1b", "gemma4:latest"]}
    except Exception as e:
        print(f"Error fetching Ollama models: {e}")
        return {"models": ["llama3.2:1b", "gemma4:latest"]}

@app.get("/api/query")
def api_query(
    query: str = Query(..., description="User search query"),
    model: str = Query(None, description="Ollama model override")
):
    def event_generator():
        try:
            for step in stream_query_rag(query, model_name=model):
                node_name = list(step.keys())[0]
                state = step[node_name]
                
                # Send SSE progress update
                payload = {
                    "node": node_name,
                    "agent_logs": state.get("agent_logs", []),
                    "answer": state.get("answer", ""),
                    "thinking": state.get("thinking", ""),
                    "citations": state.get("citations", {}),
                    "verification": state.get("verification", {}),
                    "errors": state.get("errors", [])
                }
                yield f"data: {json.dumps(payload)}\n\n"
        except Exception as e:
            err_payload = {"error": str(e)}
            yield f"data: {json.dumps(err_payload)}\n\n"
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/okf/export")
def export_okf_bundle():
    import tempfile
    import shutil
    import zipfile
    from fastapi.responses import FileResponse
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM papers ORDER BY ingested_at DESC")
    papers = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    temp_dir = tempfile.mkdtemp()
    bundle_dir = os.path.join(temp_dir, "antigravity_okf_bundle")
    papers_dir = os.path.join(bundle_dir, "papers")
    os.makedirs(papers_dir, exist_ok=True)
    
    index_content = """# Antigravity Open Knowledge Bundle

This bundle contains curated metadata and index catalogs for ingested research papers, formatted in Google's **Open Knowledge Format (OKF)**.

## Index of Ingested Papers
"""
    
    for paper in papers:
        paper_id = paper["paper_id"]
        title = paper["title"]
        authors = paper.get("authors", "Unknown")
        year = paper.get("year", "")
        url = paper.get("url", "")
        abstract = paper.get("abstract", "")
        ingested_at = paper.get("ingested_at", "")
        
        index_content += f"- [{title}](./papers/{paper_id}.md) ({year}) - *By {authors}*\n"
        
        concept_yaml = f"""---
type: "Research Paper"
title: {json.dumps(title)}
description: {json.dumps(abstract[:120] + '...' if abstract else 'No abstract available')}
resource: {json.dumps(url)}
tags: ["rag-ingested", {json.dumps(paper["source"])}]
timestamp: {json.dumps(str(ingested_at))}
authors: {json.dumps(authors)}
year: {year}
---

# {title}

**Authors:** {authors}  
**Year:** {year}  
**Source:** {paper["source"]}  
**Resource URL:** [{url}]({url})

## Abstract
{abstract}

[📄 Open Local PDF Document](http://localhost:8502/pdf/{paper_id})
"""
        with open(os.path.join(papers_dir, f"{paper_id}.md"), "w", encoding="utf-8") as f:
            f.write(concept_yaml)
            
    with open(os.path.join(bundle_dir, "index.md"), "w", encoding="utf-8") as f:
        f.write(index_content)
        
    zip_path = os.path.join(temp_dir, "antigravity_okf_bundle.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(bundle_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, bundle_dir)
                zipf.write(file_path, arcname)
                
    return FileResponse(zip_path, media_type="application/zip", filename="antigravity_okf_bundle.zip")

# Serve React static assets in production
try:
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse
    
    if os.path.exists("frontend/dist"):
        app.mount("/assets", StaticFiles(directory="frontend/dist/assets"), name="assets")
        
        @app.get("/{catchall:path}")
        def serve_react(catchall: str):
            if catchall.startswith("pdf/") or catchall.startswith("api/"):
                raise HTTPException(status_code=404, detail="Not Found")
            return FileResponse("frontend/dist/index.html")
except Exception as e:
    print(f"Warning: Static files mounting failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8502)
