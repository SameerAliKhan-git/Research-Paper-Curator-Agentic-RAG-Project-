import uvicorn
from fastapi import FastAPI, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.antigravity_rag.db_sqlite import get_paper_pdf

app = FastAPI(title="Antigravity PDF Server")

# Enable CORS so the Streamlit frontend can interact with this server cleanly
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
        raise HTTPException(status_code=404, detail=f"PDF not found for paper_id: {paper_id}")
    
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"inline; filename={paper_id}.pdf",
            "Access-Control-Allow-Origin": "*",
            "Content-Security-Policy": "frame-ancestors *"
        }
    )

def start_pdf_server(port: int = 8502):
    print(f"Starting lightweight PDF server on http://localhost:{port}...")
    uvicorn.run(app, host="localhost", port=port, log_level="error")

if __name__ == "__main__":
    start_pdf_server()
