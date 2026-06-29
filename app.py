import os
import json
import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime

# Initialize directories & config
from src.antigravity_rag.config_parser import get_config
from src.antigravity_rag.db_sqlite import init_db, get_all_papers, get_db_connection
from src.antigravity_rag.db_qdrant import init_qdrant
from src.antigravity_rag.supervisor import run_query_rag

# Set page config
st.set_page_config(
    page_title="Project Antigravity - Agentic RAG",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Inject custom CSS for premium styling (Warm cream dark theme)
st.markdown("""
<style>
    /* Styling Streamlit app */
    .stApp {
        background-color: #0E1117;
        color: #ECEFF4;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #1A1F2C;
        border-right: 1px solid #2D3748;
    }
    
    /* Heading styling */
    h1, h2, h3 {
        color: #FF5A00 !important;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Styled widgets */
    .stButton>button {
        background-color: #FF5A00;
        color: white;
        border: none;
        border-radius: 4px;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #E04E00;
        box-shadow: 0 0 10px rgba(255, 90, 0, 0.4);
    }
    
    /* Expander card background */
    .streamlit-expanderHeader {
        background-color: #1E2530;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

def generate_iframe_html(answer_text: str, citation_map: dict, is_dark: bool) -> str:
    # Color schema based on dark/light setting
    if is_dark:
        bg_color = "#0F141C"
        text_color = "#E2E8F0"
        badge_bg = "#2C5282"
        badge_color = "#EBF8FF"
        badge_border = "#3182CE"
        badge_hover_bg = "#2B6CB0"
        card_bg = "#1A202C"
        card_text = "#EDF2F7"
        card_border = "#4A5568"
        header_color = "#FFA500"
        authors_color = "#A0AEC0"
        excerpt_bg = "#111622"
        accent_color = "#FFA500"
        close_btn_color = "#E53E3E"
    else:
        bg_color = "#FFFFFF"
        text_color = "#2D3748"
        badge_bg = "#EBF8FF"
        badge_color = "#2B6CB0"
        badge_border = "#BEE3F8"
        badge_hover_bg = "#BEE3F8"
        card_bg = "#EDF2F7"
        card_text = "#1A202C"
        card_border = "#CBD5E0"
        header_color = "#FF5A00"
        authors_color = "#4A5568"
        excerpt_bg = "#F7FAFC"
        accent_color = "#FF5A00"
        close_btn_color = "#E53E3E"

    json_citation_map = json.dumps(citation_map)
    
    # Simple formatting of markdown-like elements in answer text (since we render inside iframe)
    # Convert double newlines to paragraph tags
    formatted_answer = answer_text.replace("\n\n", "</p><p>").replace("\n", "<br>")
    formatted_answer = f"<p>{formatted_answer}</p>"
    
    # Render final HTML structure
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                color: {text_color};
                background-color: {bg_color};
                margin: 0;
                padding: 10px;
                font-size: 14.5px;
                line-height: 1.6;
            }}
            .citation-badge {{
                background-color: {badge_bg};
                color: {badge_color};
                padding: 1px 5px;
                border-radius: 4px;
                font-size: 11px;
                font-weight: bold;
                cursor: pointer;
                margin-left: 2px;
                border: 1px solid {badge_border};
                user-select: none;
                transition: all 0.2s ease;
                display: inline-block;
                line-height: 1.2;
            }}
            .citation-badge:hover {{
                background-color: {badge_hover_bg};
            }}
            .floating-card {{
                display: none;
                position: fixed;
                bottom: 10px;
                left: 10px;
                right: 10px;
                background-color: {card_bg};
                color: {card_text};
                border: 1px solid {card_border};
                border-radius: 8px;
                padding: 12px;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
                z-index: 1000;
                max-height: 160px;
                overflow-y: auto;
                font-size: 12.5px;
                animation: fadeIn 0.2s ease;
            }}
            @keyframes fadeIn {{
                from {{ opacity: 0; transform: translateY(10px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            .card-header {{
                font-weight: bold;
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
                margin-bottom: 6px;
                font-size: 11.5px;
            }}
            .card-excerpt {{
                background-color: {excerpt_bg};
                border-left: 3px solid {accent_color};
                padding: 6px 10px;
                font-family: inherit;
                font-size: 12px;
                white-space: pre-wrap;
            }}
            .close-btn {{
                cursor: pointer;
                font-weight: bold;
                font-size: 18px;
                color: {close_btn_color};
                padding: 0 5px;
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
st.markdown("# 💬 Automated Research QA")
st.markdown("Ask natural-language questions and synthesize answers grounded in academic literature.")

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
                            st.markdown(f"**[{cite_num}]** {title}{author_str}")
                            
                            # PDF or URL link
                            if details.get("full_text_path") and os.path.exists(details["full_text_path"]):
                                pdf_path = details["full_text_path"]
                                with open(pdf_path, "rb") as f:
                                    st.download_button(
                                        label=f"💾 Download {os.path.basename(pdf_path)}",
                                        data=f,
                                        file_name=os.path.basename(pdf_path),
                                        mime="application/pdf",
                                        key=f"dl_{msg_idx}_{cite_num}"
                                    )
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
        with st.spinner("Agentic RAG pipeline executing..."):
            try:
                # Run the LangGraph supervisor workflow
                response = run_query_rag(user_query)
                
                answer = response["answer"]
                citations = response["citations"]
                
                # Render answer in HTML
                html_payload = generate_iframe_html(answer, citations, is_dark)
                components.html(html_payload, height=260, scrolling=True)
                
                # Save message in session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "citations": citations,
                    "query": user_query
                })
                st.rerun()
                
            except Exception as e:
                st.error(f"Failed to process query: {e}")
