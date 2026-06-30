import os
import json
import socket
import threading
import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime

# Initialize directories & config
from src.antigravity_rag.config_parser import get_config
from src.antigravity_rag.db_sqlite import init_db, get_all_papers, get_db_connection, get_paper_chunks
from src.antigravity_rag.db_qdrant import init_qdrant
from src.antigravity_rag.supervisor import run_query_rag, stream_query_rag
from src.antigravity_rag.pdf_server import start_pdf_server

# Set page config
st.set_page_config(
    page_title="Project Antigravity - Agentic RAG",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Start lightweight PDF server on port 8502 in a daemon thread if free
if "pdf_server_started" not in st.session_state:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("localhost", 8502))
        s.close()
        port_free = True
    except OSError:
        port_free = False
        
    if port_free:
        t = threading.Thread(target=start_pdf_server, args=(8502,), daemon=True)
        t.start()
    st.session_state.pdf_server_started = True

# Initialize database schemas
init_db()
init_qdrant()

# Load config
config = get_config()

# Local feedback file path
FEEDBACK_FILE = "feedback.json"

def save_feedback(query: str, rating: str, answer: str):
    feedback_data = []
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
                feedback_data = json.load(f)
        except Exception:
            pass
            
    feedback_data.append({
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "rating": rating,
        "answer": answer
    })
    
    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump(feedback_data, f, indent=2)

def get_today_ingestion_count() -> int:
    conn = get_db_connection()
    cursor = conn.cursor()
    # Count papers ingested in the last 24h
    yesterday = (datetime.now() - timedelta(days=1)).isoformat()
    try:
        cursor.execute("SELECT COUNT(*) as count FROM papers WHERE ingested_at >= ?", (yesterday,))
        count = cursor.fetchone()["count"]
    except Exception:
        cursor.execute("SELECT COUNT(*) as count FROM papers")
        count = cursor.fetchone()["count"]
    conn.close()
    return count

from datetime import timedelta

# Inject custom CSS for premium styling (Cyberpunk glassmorphic dark theme)
st.markdown("""
<style>
    /* Import modern typography */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global app container background */
    .stApp {
        background: radial-gradient(circle at 80% 20%, #151A2E 0%, #0B0E17 100%) !important;
        color: #ECEFF4 !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Sidebar premium redesign */
    section[data-testid="stSidebar"] {
        background-color: rgba(11, 14, 23, 0.95) !important;
        border-right: 1px solid rgba(255, 90, 0, 0.15) !important;
        backdrop-filter: blur(12px);
    }
    
    section[data-testid="stSidebar"] h1 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #FF6B00 0%, #E04E00 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 26px !important;
        letter-spacing: -0.5px !important;
        margin-bottom: 2px !important;
    }
    
    /* Native Chat bubble custom overrides */
    div[data-testid="stChatMessage"] {
        background-color: rgba(22, 29, 43, 0.45) !important;
        border: 1px solid rgba(255, 255, 255, 0.03) !important;
        border-radius: 16px !important;
        padding: 16px 20px !important;
        margin-bottom: 16px !important;
        box-shadow: 0 4px 25px rgba(0, 0, 0, 0.2) !important;
        backdrop-filter: blur(8px);
    }
    
    /* Distinguish User messages */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
        background: linear-gradient(135deg, rgba(255, 107, 0, 0.08) 0%, rgba(255, 107, 0, 0.02) 100%) !important;
        border: 1px solid rgba(255, 107, 0, 0.15) !important;
        box-shadow: 0 4px 15px rgba(255, 107, 0, 0.04) !important;
    }
    
    /* Distinguish Assistant messages */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
        background-color: rgba(20, 26, 38, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
    }
    
    /* Styled Chat Input box */
    div[data-testid="stChatInput"] textarea {
        background-color: #121824 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        color: #ECEFF4 !important;
        border-radius: 20px !important;
        padding: 12px 16px !important;
        font-size: 14.5px !important;
        transition: border-color 0.2s ease !important;
    }
    div[data-testid="stChatInput"] textarea:focus {
        border-color: rgba(255, 90, 0, 0.5) !important;
    }
    div[data-testid="stChatInput"] {
        background-color: transparent !important;
        border: none !important;
    }
    
    /* Heading typography styling */
    h1, h2, h3 {
        color: #FF5A00 !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px !important;
    }
    
    /* Premium button designs */
    .stButton>button {
        background: linear-gradient(135deg, #FF6B00 0%, #E04E00 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(255, 107, 0, 0.25) !important;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(255, 107, 0, 0.45) !important;
    }
    
    /* Tabs redesign */
    button[data-baseweb="tab"] {
        font-family: 'Outfit', sans-serif !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        color: #718096 !important;
        background-color: transparent !important;
        border: none !important;
        padding: 10px 16px !important;
        transition: all 0.2s ease !important;
    }
    button[data-baseweb="tab"]:hover {
        color: #FF5A00 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #FF5A00 !important;
        border-bottom: 2px solid #FF5A00 !important;
    }
    
    /* Expander card overrides */
    .streamlit-expanderHeader {
        background-color: rgba(22, 29, 43, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.03) !important;
        border-radius: 8px !important;
    }
    
    /* Metrics display styling */
    div[data-testid="stMetricValue"] {
        color: #FF6B00 !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)

def generate_iframe_html(answer_text: str, citation_map: dict, is_dark: bool) -> str:
    import re
    # Color schema based on theme setting
    bg_color = "transparent"
    text_color = "#ECEFF4" if is_dark else "#2D3748"
    card_bg = "rgba(30, 37, 54, 0.95)" if is_dark else "rgba(237, 242, 247, 0.95)"
    card_text = "#ECEFF4" if is_dark else "#1A202C"
    card_border = "rgba(255, 90, 0, 0.25)" if is_dark else "rgba(255, 90, 0, 0.15)"
    header_color = "#FF6B00"
    authors_color = "#A0AEC0" if is_dark else "#4A5568"
    excerpt_bg = "rgba(13, 17, 28, 0.6)" if is_dark else "rgba(255, 255, 255, 0.8)"
    accent_color = "#FF6B00"
    close_btn_color = "#E53E3E"

    json_citation_map = json.dumps(citation_map)
    
    # Process simple markdown formats in answer text for iframe display
    formatted_answer = answer_text
    # Bold text
    formatted_answer = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', formatted_answer)
    # Inline code
    formatted_answer = re.sub(r'`(.*?)`', r'<code style="background-color: rgba(255,255,255,0.08); padding: 2px 4px; border-radius: 4px; font-family: monospace; font-size: 0.9em; color: #FFA500;">\1</code>', formatted_answer)
    
    # Convert newlines
    formatted_answer = formatted_answer.replace("\n\n", "</p><p>").replace("\n", "<br>")
    formatted_answer = f"<p>{formatted_answer}</p>"
    
    # Render final HTML structure
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@600;700&family=Inter:wght@400;500;600&display=swap');
            
            body {{
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                color: {text_color};
                background-color: {bg_color};
                margin: 0;
                padding: 10px;
                font-size: 14.5px;
                line-height: 1.62;
            }}
            .citation-badge {{
                background: rgba(255, 107, 0, 0.12);
                color: #FF6B00;
                padding: 2px 6px;
                border-radius: 6px;
                font-size: 11px;
                font-weight: 600;
                cursor: pointer;
                margin-left: 2px;
                border: 1px solid rgba(255, 107, 0, 0.3);
                user-select: none;
                transition: all 0.2s cubic-bezier(0.25, 0.8, 0.25, 1);
                display: inline-block;
                line-height: 1;
            }}
            .citation-badge:hover {{
                background: rgba(255, 107, 0, 0.25);
                transform: scale(1.05);
                box-shadow: 0 0 8px rgba(255, 107, 0, 0.4);
            }}
            .citation-badge a {{
                color: inherit;
                text-decoration: none;
            }}
            .floating-card {{
                display: none;
                position: fixed;
                bottom: 12px;
                left: 12px;
                right: 12px;
                background: {card_bg};
                backdrop-filter: blur(12px);
                color: {card_text};
                border: 1px solid {card_border};
                border-radius: 12px;
                padding: 14px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.45);
                z-index: 1000;
                max-height: 155px;
                overflow-y: auto;
                font-size: 12.5px;
                animation: slideUp 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
            }}
            @keyframes slideUp {{
                from {{ opacity: 0; transform: translateY(18px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            .card-header {{
                font-family: 'Outfit', sans-serif;
                font-weight: 700;
                font-size: 13.5px;
                margin-bottom: 4px;
                color: {header_color};
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .card-authors {{
                font-style: italic;
                color: {authors_color};
                margin-bottom: 8px;
                font-size: 11.5px;
            }}
            .card-excerpt {{
                background: {excerpt_bg};
                border-left: 3px solid {accent_color};
                padding: 8px 12px;
                border-radius: 0 6px 6px 0;
                font-family: inherit;
                font-size: 12px;
                line-height: 1.5;
                white-space: pre-wrap;
                color: {text_color};
            }}
            .close-btn {{
                cursor: pointer;
                font-weight: bold;
                font-size: 18px;
                color: {close_btn_color};
                padding: 0 5px;
                transition: color 0.2s ease;
            }}
            .close-btn:hover {{
                color: #FC8181;
            }}
            
            /* Scrollbars */
            ::-webkit-scrollbar {{
                width: 6px;
            }}
            ::-webkit-scrollbar-track {{
                background: rgba(255, 255, 255, 0.02);
            }}
            ::-webkit-scrollbar-thumb {{
                background: rgba(255, 90, 0, 0.25);
                border-radius: 3px;
            }}
            ::-webkit-scrollbar-thumb:hover {{
                background: rgba(255, 90, 0, 0.45);
            }}
        </style>
    </head>
    <body>
        <div id="content">
            {formatted_answer}
        </div>
        <div id="floating-card" class="floating-card">
            <div class="card-header">
                <span id="card-title">Paper Title</span>
                <span class="close-btn" onclick="closeCard()">×</span>
            </div>
            <div id="card-authors" class="card-authors">Authors</div>
            <div id="card-excerpt" class="card-excerpt">Excerpt...</div>
        </div>

        <script>
            const citationData = {json_citation_map};
            
            function showCitation(id) {{
                const data = citationData[id];
                if (data) {{
                    document.getElementById('card-title').innerText = data.paper_title + " (" + data.year + ")";
                    document.getElementById('card-authors').innerText = "By: " + data.authors;
                    document.getElementById('card-excerpt').innerText = data.excerpt;
                    document.getElementById('floating-card').style.display = 'block';
                }}
            }}
            
            function closeCard() {{
                document.getElementById('floating-card').style.display = 'none';
            }}
        </script>
    </body>
    </html>
    """
    return html_content

# Sidebar UI
st.sidebar.markdown("# 🌌 Project Antigravity")
st.sidebar.markdown("*Local Agentic RAG Platform*")
st.sidebar.markdown("---")

# Theme Selection
theme_selection = st.sidebar.radio("Theme Mode", ["Dark", "Light"], index=0)
is_dark = (theme_selection == "Dark")

# Placeholder for Agent Status Dashboard
st.sidebar.markdown("### 🤖 Agent Status Dashboard")
agent_status_placeholder = st.sidebar.empty()

def render_agent_status(active_agent: str):
    agents = {
        "Supervisor": "check_knowledge",
        "Web Search": "web_search",
        "PDF Ingestion": "download_and_index",
        "Retrieval": "retrieve",
        "Generation": "generate",
        "Verification": "verify"
    }
    
    html = "<div style='background-color: #1A1F2C; padding: 12px; border-radius: 8px; border: 1px solid #2D3748;'>"
    for name, node in agents.items():
        if active_agent == node:
            status_text = "WORKING"
            bg = "#FF5A00"
            color = "#FFFFFF"
        elif active_agent == "Idle":
            status_text = "IDLE"
            bg = "#2D3748"
            color = "#A0AEC0"
        else:
            status_text = "STANDBY"
            bg = "#111622"
            color = "#718096"
            
        html += f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; font-family: sans-serif; font-size: 13px;"><span style="font-weight: bold; color: #ECEFF4;">{name}</span><span style="background-color: {bg}; color: {color}; padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: bold; letter-spacing: 0.5px;">{status_text}</span></div>'
    html += "</div>"
    agent_status_placeholder.markdown(html, unsafe_allow_html=True)

# Render initial state
render_agent_status("Idle")

st.sidebar.markdown("### ⚙️ Search Settings")
max_search_papers = st.sidebar.slider("Web Search Max Papers", 1, 10, 5)
max_download_time = st.sidebar.slider("Max Download Timeout (s)", 2, 15, 5)
llm_temperature = st.sidebar.slider("LLM Temperature", 0.0, 1.0, 0.2)

st.sidebar.markdown("### 📡 Source Filters")
filter_arxiv = st.sidebar.checkbox("ArXiv", value=True)
filter_ss = st.sidebar.checkbox("Semantic Scholar", value=True)
filter_gs = st.sidebar.checkbox("Google Scholar", value=False)

# Update config dynamically in memory
config.sources["arxiv"]["enabled"] = filter_arxiv
config.sources["arxiv"]["max_results"] = max_search_papers
config.sources["semantic_scholar"]["enabled"] = filter_ss
config.sources["semantic_scholar"]["max_results"] = max_search_papers
config.sources["google_scholar"]["enabled"] = filter_gs
config.llm["temperature"] = llm_temperature

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Today's Dashboard")

# Get today's stats
today_count = get_today_ingestion_count()
st.sidebar.metric("Papers Ingested (Last 24h)", today_count)

# Show last ingested paper title
papers_list = get_all_papers()
if papers_list:
    last_paper = papers_list[0]
    st.sidebar.markdown(f"**Last Ingested:**\n*{last_paper['title']}*")
else:
    st.sidebar.markdown("**No papers indexed yet.**")

# Manual Ingestion trigger
if st.sidebar.button("⚡ Run Scheduled Ingestion Now"):
    with st.spinner("Executing background daily ingestion..."):
        try:
            from ingest_daily import run_ingestion
            run_ingestion()
            st.sidebar.success("Ingestion finished!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Ingestion failed: {e}")

# Main Chat Layout
st.markdown("# 🌌 Antigravity Research Assistant")

# Tabs for Chat vs. Source Explorer
tab_chat, tab_explorer = st.tabs(["💬 Assistant Chat", "🔍 Deep Source Explorer"])

with tab_chat:
    # Chat History Init
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display Chat History
    for msg_idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                # Render assistant's custom HTML
                html_payload = generate_iframe_html(
                    message["content"], 
                    message.get("citations", {}), 
                    is_dark
                )
                components.html(html_payload, height=260, scrolling=True)
                
                # Citations and Sources expander
                if message.get("citations"):
                    with st.expander("📚 Citations & Sources", expanded=False):
                        seen_titles = set()
                        for cite_num, details in message["citations"].items():
                            title = details["paper_title"]
                            if title not in seen_titles:
                                seen_titles.add(title)
                                author_str = f" (*{details.get('authors', 'Unknown')}*)" if details.get('authors') else ""
                                
                                # Check NLI verification details
                                ver_info = message.get("verification", {}).get(details["chunk_id"])
                                ver_warning = ""
                                if ver_info:
                                    supported = ver_info.get("supported", True)
                                    confidence = ver_info.get("confidence", 1.0)
                                    if not supported or confidence < 0.5:
                                        ver_warning = f" ⚠️ *[Refuted/Low NLI Confidence ({confidence:.2f})]*"
                                
                                st.markdown(f"**[{cite_num}]** {title}{author_str} {ver_warning}")
                                
                                # Database PDF link
                                paper_id = details.get("paper_id")
                                page_num = details.get("page", 1)
                                if paper_id:
                                    pdf_url = f"http://localhost:8502/pdf/{paper_id}?page={page_num}"
                                    st.markdown(f"🔗 [View PDF Source ({details.get('section_title', f'Page {page_num}')})]({pdf_url})")
                                elif details.get("url"):
                                    st.markdown(f"🔗 [Publisher Link]({details['url']})")
                                    
                # Feedback component
                col1, col2, col3 = st.columns([1, 1, 15])
                with col1:
                    if st.button("👍", key=f"up_{msg_idx}"):
                        save_feedback(message.get("query", ""), "up", message["content"])
                        st.toast("Thank you for your feedback!", icon="🎉")
                with col2:
                    if st.button("👎", key=f"down_{msg_idx}"):
                        save_feedback(message.get("query", ""), "down", message["content"])
                        st.toast("Feedback recorded.", icon="📝")

    # User Input Box
    if user_query := st.chat_input("Ask a research question..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Generate response
        with st.chat_message("assistant"):
            # Progress thinking box
            thinking_container = st.container()
            with thinking_container:
                st.markdown("### 🧠 Thinking Process")
                log_box = st.empty()
                
            try:
                # Run stream
                final_state = None
                for event in stream_query_rag(user_query):
                    node_name = list(event.keys())[0]
                    final_state = event[node_name]
                    
                    # Update Agent Dashboard status dynamically
                    render_agent_status(node_name)
                    
                    # Update live log box
                    logs = final_state.get("agent_logs", [])
                    log_html = "<div style='max-height: 180px; overflow-y: auto; background-color: #1A1F2C; padding: 10px; border-radius: 5px; font-family: monospace; font-size: 12px; color: #ECEFF4;'>"
                    for log in logs:
                        log_html += f"<div style='margin-bottom: 4px;'>{log}</div>"
                    log_html += "</div>"
                    log_box.markdown(log_html, unsafe_allow_html=True)
                
                # Reset dashboard back to Idle
                render_agent_status("Idle")
                
                # Clear thinking container
                thinking_container.empty()
                
                answer = final_state["answer"]
                citations = final_state["citations"]
                verification = final_state.get("verification", {})
                
                # Render answer in HTML
                html_payload = generate_iframe_html(answer, citations, is_dark)
                components.html(html_payload, height=260, scrolling=True)
                
                # Save message in session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "citations": citations,
                    "verification": verification,
                    "query": user_query
                })
                st.rerun()
                
            except Exception as e:
                render_agent_status("Idle")
                st.error(f"Failed to process query: {e}")

with tab_explorer:
    st.markdown("### 🔍 Deep Source Explorer")
    st.markdown("Explore extracted text chunks and high-resolution PDF pages side-by-side.")
    
    papers = get_all_papers()
    if not papers:
        st.info("No research papers ingested yet. Ask a question or run ingestion to download papers.")
    else:
        # Paper selection dropdown
        paper_options = {p["title"]: p for p in papers}
        selected_title = st.selectbox("Select a Research Paper to Explore", list(paper_options.keys()))
        selected_paper = paper_options[selected_title]
        paper_id = selected_paper["paper_id"]
        
        # Columns
        col_text, col_pdf = st.columns([1, 1])
        
        with col_text:
            st.markdown("#### 📄 Extracted Text Chunks")
            chunks = get_paper_chunks(paper_id)
            if not chunks:
                st.write("*Abstract:*")
                st.write(selected_paper["abstract"])
            else:
                for chunk in chunks:
                    sec_title = chunk.get("section_title") or f"Chunk {chunk['chunk_index']}"
                    page_lbl = f" (Page {chunk['page_number']})" if chunk.get("page_number") else ""
                    with st.expander(f"📍 {sec_title}{page_lbl}", expanded=False):
                        st.write(chunk["chunk_text"])
                        st.caption(f"Index: {chunk['chunk_index']} | Tokens: {chunk['token_count']}")
                        
        with col_pdf:
            st.markdown("#### 📜 PDF Viewer")
            pdf_url = f"http://localhost:8502/pdf/{paper_id}"
            st.markdown(f"🔗 [Open PDF in new tab]({pdf_url})")
            
            # Embed PDF using iframe pointing to our FastAPI PDF server
            pdf_embed_html = f"""
            <iframe src="{pdf_url}" width="100%" height="700px" style="border: 1px solid #4A5568; border-radius: 8px;"></iframe>
            """
            components.html(pdf_embed_html, height=720)
