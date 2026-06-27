import json
import logging
import math
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Iterator, List

import gradio as gr
import httpx
import src.config

# Patch the database URL environment variable to point to localhost instead of postgres host when running on host machine
db_url = os.environ.get("POSTGRES_DATABASE_URL") or "postgresql+psycopg2://rag_user:rag_password@postgres:5432/rag_db"
if "postgres:5432" in db_url:
    os.environ["POSTGRES_DATABASE_URL"] = db_url.replace("postgres:5432", "localhost:5432")
    src.config._settings_cache = None

from src.config import get_settings
from src.db.factory import make_database
from src.models.researcher import DailyBriefing, ResearcherInterest

logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama3.2:1b")
AVAILABLE_CATEGORIES = ["cs.AI", "cs.LG", "cs.CL", "cs.CV"]


def format_source_link(source) -> str:
    """Format a source item (either string or dict) into a markdown link."""
    if isinstance(source, dict):
        url = source.get("url", "#")
        title = source.get("title") or source.get("arxiv_id") or "Source Paper"
        if url == "#" or not url:
            return f"Uploaded Paper: {title}"
        return f"[{title}]({url})"
    else:
        url = str(source)
        title = url.split("/")[-1] if "/" in url else "Source Link"
        if url == "#" or not url:
            return "Uploaded Paper"
        return f"[{title}]({url})"


async def stream_response(
    query: str, top_k: int = 3, use_hybrid: bool = True, model: str = DEFAULT_MODEL, categories: str = ""
) -> Iterator[str]:
    """Stream response from the RAG API"""
    if not query.strip():
        yield "Please enter a question."
        return

    # Parse categories
    category_list = [cat.strip() for cat in categories.split(",") if cat.strip()] if categories else None

    # Prepare request payload
    payload = {"query": query, "top_k": top_k, "use_hybrid": use_hybrid, "model": model, "categories": category_list}

    try:
        url = f"{API_BASE_URL}/stream"
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", url, json=payload, headers={"Accept": "text/plain"}) as response:
                if response.status_code != 200:
                    yield f"Error: API returned status {response.status_code}"
                    return

                current_answer = ""
                sources = []
                chunks_used = 0
                search_mode = ""

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix
                        try:
                            data = json.loads(data_str)

                            # Handle error
                            if "error" in data:
                                yield f"Error: {data['error']}"
                                return

                            # Handle metadata
                            if "sources" in data:
                                sources = data["sources"]
                                chunks_used = data.get("chunks_used", 0)
                                search_mode = data.get("search_mode", "unknown")
                                continue

                            # Handle streaming chunks
                            if "chunk" in data:
                                current_answer += data["chunk"]
                                formatted_response = current_answer
                                if sources or chunks_used:
                                    formatted_response += f"\n\n**Search Info:**\n"
                                    formatted_response += f"- Mode: {search_mode}\n"
                                    formatted_response += f"- Chunks used: {chunks_used}\n"
                                    if sources:
                                        formatted_response += f"- Sources: {len(sources)} papers\n"
                                        for i, source in enumerate(sources[:3], 1):
                                            formatted_response += f"  {i}. {format_source_link(source)}\n"
                                        if len(sources) > 3:
                                            formatted_response += f"  ... and {len(sources) - 3} more\n"

                                yield formatted_response

                            # Handle completion
                            if data.get("done", False):
                                final_answer = data.get("answer", current_answer)
                                if final_answer != current_answer:
                                    current_answer = final_answer

                                formatted_response = current_answer
                                if sources or chunks_used:
                                    formatted_response += f"\n\n**Search Info:**\n"
                                    formatted_response += f"- Mode: {search_mode}\n"
                                    formatted_response += f"- Chunks used: {chunks_used}\n"
                                    if sources:
                                        formatted_response += f"- Sources: {len(sources)} papers\n"
                                        for i, source in enumerate(sources[:3], 1):
                                            formatted_response += f"  {i}. {format_source_link(source)}\n"
                                        if len(sources) > 3:
                                            formatted_response += f"  ... and {len(sources) - 3} more\n"

                                yield formatted_response
                                break

                        except json.JSONDecodeError:
                            continue

    except httpx.RequestError as e:
        yield f"Connection error: {str(e)}\nMake sure the API server is running at {API_BASE_URL}"
    except Exception as e:
        yield f"Unexpected error: {str(e)}"


# Database Helpers for Tab 3 (Briefings & Interests)
def get_interests() -> List[str]:
    try:
        db = make_database()
        with db.get_session() as session:
            interests = session.query(ResearcherInterest).order_by(ResearcherInterest.keyword).all()
            return [i.keyword for i in interests]
    except Exception as e:
        logger.error(f"Failed to get interests: {e}")
        return []


def add_interest(keyword: str) -> str:
    keyword = keyword.strip()
    if not keyword:
        return "Keyword cannot be empty."
    try:
        db = make_database()
        with db.get_session() as session:
            existing = session.query(ResearcherInterest).filter_by(keyword=keyword).first()
            if existing:
                return f"Keyword '{keyword}' already exists."
            interest = ResearcherInterest(keyword=keyword)
            session.add(interest)
            session.commit()
        return f"Successfully added keyword '{keyword}'."
    except Exception as e:
        logger.error(f"Failed to add interest: {e}")
        return f"Error: {e}"


def remove_interest(keyword: str) -> str:
    keyword = keyword.strip()
    if not keyword:
        return "Keyword cannot be empty."
    try:
        db = make_database()
        with db.get_session() as session:
            interest = session.query(ResearcherInterest).filter_by(keyword=keyword).first()
            if not interest:
                return f"Keyword '{keyword}' not found."
            session.delete(interest)
            session.commit()
        return f"Successfully removed keyword '{keyword}'."
    except Exception as e:
        logger.error(f"Failed to remove interest: {e}")
        return f"Error: {e}"


def get_briefings_md() -> str:
    try:
        db = make_database()
        with db.get_session() as session:
            briefings = session.query(DailyBriefing).order_by(DailyBriefing.created_at.desc()).limit(20).all()
            if not briefings:
                return "No daily briefings available yet. The scheduled Airflow daily_arxiv_briefing DAG generates these."

            md = ""
            for b in briefings:
                date_str = b.published_date.strftime("%Y-%m-%d") if b.published_date else "Unknown Date"
                score_str = f"{b.score:.2f}" if b.score else "N/A"
                arxiv_url = f"https://arxiv.org/abs/{b.arxiv_id}"
                md += f"### [{b.title}]({arxiv_url}) (arXiv ID: {b.arxiv_id})\n"
                md += f"- **Published Date:** {date_str}\n"
                md += f"- **Relevance Score:** {score_str}\n"
                md += f"#### Technical Briefing Summary:\n{b.summary}\n\n---\n\n"
            return md
    except Exception as e:
        logger.error(f"Failed to get briefings: {e}")
        return f"Error loading briefings: {e}"


# LaTeX related work generator helper
async def compile_latex_zip(arxiv_ids_str: str, model: str) -> tuple[str, str]:
    arxiv_ids = [aid.strip() for aid in arxiv_ids_str.split(",") if aid.strip()]
    if not arxiv_ids:
        return "Error: Please enter at least one arXiv ID.", None

    payload = {"arxiv_ids": arxiv_ids, "model": model}
    try:
        url = f"{API_BASE_URL}/literature/related-work"
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json=payload)
            if response.status_code != 200:
                detail = "Unknown error"
                try:
                    detail = response.json().get("detail", detail)
                except Exception:
                    pass
                return f"Error generating related work: {detail}", None

            # Save ZIP content to a temporary directory
            temp_dir = Path("data/latex_builds")
            temp_dir.mkdir(parents=True, exist_ok=True)
            zip_path = temp_dir / "related_work_latex.zip"
            zip_path.write_bytes(response.content)

            # Extract the related_work.tex content for UI display
            with zipfile.ZipFile(zip_path, "r") as zf:
                tex_content = zf.read("related_work.tex").decode("utf-8")

            return tex_content, str(zip_path)

    except Exception as e:
        logger.error(f"Failed compile_latex_zip: {e}")
        return f"Error: {e}", None


# BibTeX fetcher helper
async def fetch_bibtex(arxiv_id: str) -> str:
    arxiv_id = arxiv_id.strip()
    if not arxiv_id:
        return "Please enter a valid arXiv ID."
    try:
        url = f"{API_BASE_URL}/papers/{arxiv_id}/bibtex"
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(url)
            if response.status_code != 200:
                detail = "Unknown error"
                try:
                    detail = response.json().get("detail", detail)
                except Exception:
                    pass
                return f"Error: {detail} (Make sure the paper is ingested first)"
            return response.json().get("bibtex", "")
    except Exception as e:
        return f"Error: {e}"


def create_gradio_interface():
    """Create and configure the new premium tabbed interface"""

    custom_css = """
    body, .gradio-container {
        background-color: #0b0f19 !important;
        font-family: 'Outfit', 'Inter', system-ui, sans-serif !important;
        color: #e2e8f0 !important;
    }
    h1, h2, h3, h4 {
        font-family: 'Outfit', sans-serif !important;
        background: linear-gradient(135deg, #ff7e5f, #feb47b) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        font-weight: 800 !important;
    }
    .custom-card {
        background-color: #151f32 !important;
        border: 1px solid #1e293b !important;
        border-radius: 16px !important;
        padding: 20px !important;
    }
    .primary-btn {
        background: linear-gradient(135deg, #ea2804, #ff5f43) !important;
        color: #ffffff !important;
        border-radius: 9999px !important;
        font-weight: 600 !important;
        border: none !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    }
    .primary-btn:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 10px 15px -3px rgba(234, 40, 4, 0.3) !important;
    }
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
        background-color: #1e293b !important;
        color: #f1f5f9 !important;
    }
    """

    with gr.Blocks(
        title="arXiv Paper Curator Console - RAG",
        theme=gr.themes.Default(primary_hue="orange", secondary_hue="slate"),
        css=custom_css,
    ) as interface:
        gr.Markdown(
            """
            # 🔬 Production-Grade Agentic RAG Console
            *Built for senior researchers, scholars, and AI engineers*
            """
        )

        with gr.Tabs():
            # TAB 1: Chat Console
            with gr.Tab("💬 RAG Chat Console"):
                with gr.Row():
                    with gr.Column(scale=3):
                        query_input = gr.Textbox(
                            label="Researcher Query",
                            placeholder="State your question, e.g., 'What are the main architectural limits of transformers?'",
                            lines=2,
                        )
                        submit_btn = gr.Button("Execute RAG Pipeline", variant="primary", size="lg", elem_classes=["primary-btn"])
                        response_output = gr.Markdown(label="Answer", value="Ask a question to get started!", height=400)

                    with gr.Column(scale=1):
                        with gr.Group(elem_classes=["custom-card"]):
                            gr.Markdown("### ⚙️ Pipeline Configurations")
                            top_k = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Chunks to Retrieve")
                            use_hybrid = gr.Checkbox(value=True, label="Hybrid Search (BM25 + Vector)")
                            model_choice = gr.Dropdown(
                                choices=["llama3.2:1b", "llama3.2:3b", "llama3.1:8b"], value=DEFAULT_MODEL, label="LLM Model"
                            )
                            categories = gr.Textbox(label="arXiv Categories", placeholder="cs.AI, cs.LG, cs.CL")

                        # Quick BibTeX Retrieval Tool
                        with gr.Group(elem_classes=["custom-card"]):
                            gr.Markdown("### 📄 Quick BibTeX Citation")
                            bibtex_arxiv_id = gr.Textbox(label="arXiv ID", placeholder="1706.03762")
                            fetch_bib_btn = gr.Button("Get BibTeX")
                            bibtex_output = gr.Code(label="BibTeX Citation", language=None, interactive=False)

                # Examples
                gr.Examples(
                    examples=[
                        ["What are transformers in machine learning?", 3, True, "llama3.2:1b", "cs.AI, cs.LG"],
                        ["How does retrieval-augmented generation prevent hallucination?", 4, True, "llama3.2:1b", "cs.AI"],
                    ],
                    inputs=[query_input, top_k, use_hybrid, model_choice, categories],
                )

            # TAB 2: LaTeX Related Work Generator
            with gr.Tab("📝 LaTeX Literature Review Synthesizer"):
                gr.Markdown(
                    """
                    ### 📂 Generate LaTeX Comparative Analysis
                    Provide a comma-separated list of ingested arXiv IDs. The agent will fetch their abstracts 
                    and synthesize a complete, professional **Related Work** section (`.tex`) and its matching bibliography (`.bib`) file.
                    """
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        latex_arxiv_ids = gr.Textbox(
                            label="arXiv IDs (comma-separated)", placeholder="1706.03762, 2005.11401, 2106.09685"
                        )
                        latex_model = gr.Dropdown(
                            choices=["llama3.2:1b", "llama3.2:3b", "llama3.1:8b"], value=DEFAULT_MODEL, label="Synthesizer Model"
                        )
                        gen_latex_btn = gr.Button("Generate Compile-Ready LaTeX", variant="primary", elem_classes=["primary-btn"])

                    with gr.Column(scale=3):
                        latex_preview = gr.Textbox(label="LaTeX Code Preview", lines=12, max_lines=25, interactive=False)
                        download_zip = gr.File(label="Download Compile-Ready ZIP (.tex + .bib)")

            # TAB 3: Daily Briefings & Profiles
            with gr.Tab("📅 Daily arXiv Briefings"):
                gr.Markdown(
                    """
                    ### 🔍 Curated Daily Preprints
                    Ranked and generated dynamically via the **Airflow scheduler** based on your specific researcher interest profile keywords.
                    """
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group(elem_classes=["custom-card"]):
                            gr.Markdown("### 🏷️ My Interest Profile")
                            current_keywords = gr.Dropdown(
                                label="Current Interest Keywords", choices=get_interests(), value=None, interactive=True
                            )
                            refresh_interests_btn = gr.Button("Refresh Profile", size="sm")

                            new_keyword = gr.Textbox(label="Add New Keyword", placeholder="e.g. 'Constitutional AI'")
                            add_keyword_btn = gr.Button("Add Keyword", variant="primary", size="sm")

                            remove_keyword_btn = gr.Button("Remove Selected Keyword", variant="stop", size="sm")

                            action_status = gr.Textbox(label="Status", interactive=False)

                    with gr.Column(scale=3):
                        gr.Markdown("### 📰 Daily Briefing Feed (Latest)")
                        briefings_display = gr.Markdown(value=get_briefings_md(), elem_classes=["response-markdown"])
                        refresh_feed_btn = gr.Button("Refresh Briefings Feed", elem_classes=["primary-btn"])

        # Chat interaction setup
        submit_btn.click(
            fn=stream_response,
            inputs=[query_input, top_k, use_hybrid, model_choice, categories],
            outputs=[response_output],
            show_progress=True,
        )
        query_input.submit(
            fn=stream_response,
            inputs=[query_input, top_k, use_hybrid, model_choice, categories],
            outputs=[response_output],
            show_progress=True,
        )

        # BibTeX button click
        fetch_bib_btn.click(fn=fetch_bibtex, inputs=[bibtex_arxiv_id], outputs=[bibtex_output])

        # LaTeX button click
        gen_latex_btn.click(
            fn=compile_latex_zip,
            inputs=[latex_arxiv_ids, latex_model],
            outputs=[latex_preview, download_zip],
            show_progress=True,
        )

        # Interest profile managers
        def add_interest_callback(kw):
            status = add_interest(kw)
            return status, gr.Dropdown(choices=get_interests(), value=None)

        def remove_interest_callback(kw):
            status = remove_interest(kw)
            return status, gr.Dropdown(choices=get_interests(), value=None)

        def refresh_interests_callback():
            return gr.Dropdown(choices=get_interests(), value=None)

        add_keyword_btn.click(fn=add_interest_callback, inputs=[new_keyword], outputs=[action_status, current_keywords])

        remove_keyword_btn.click(
            fn=remove_interest_callback, inputs=[current_keywords], outputs=[action_status, current_keywords]
        )

        refresh_interests_btn.click(fn=refresh_interests_callback, outputs=[current_keywords])

        refresh_feed_btn.click(fn=get_briefings_md, outputs=[briefings_display])

        gr.Markdown(
            """
            ---
            *Powered by Redis cache, Jina Embeddings, OpenSearch hybrid scoring, and local Ollama inference.*
            """
        )

    return interface


def main():
    import os

    print("🚀 Starting Production-Grade arXiv RAG Gradio Interface...")
    print(f"📡 API Base URL: {API_BASE_URL}")

    username = os.environ.get("GRADIO_USERNAME")
    password = os.environ.get("GRADIO_PASSWORD")

    interface = create_gradio_interface()

    launch_kwargs = {
        "server_name": "0.0.0.0",
        "server_port": 7861,
        "share": False,
        "show_error": True,
        "quiet": False,
    }

    if username and password:
        print(f"🔒 Gradio authentication enabled (User: {username})")
        launch_kwargs["auth"] = (username, password)
    else:
        print("⚠️ Gradio authentication disabled. Set GRADIO_USERNAME and GRADIO_PASSWORD env vars to secure.")

    interface.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
