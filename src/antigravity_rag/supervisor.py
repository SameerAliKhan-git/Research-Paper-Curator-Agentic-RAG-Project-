from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END, START

from src.antigravity_rag.config_parser import get_config
from src.antigravity_rag.db_sqlite import get_db_connection
from src.antigravity_rag.db_qdrant import search_vectors
from src.antigravity_rag.local_embeddings import embed_text
from src.antigravity_rag.search_engine import hybrid_search
from src.antigravity_rag.ollama_chat import generate_answer
from src.antigravity_rag.ingestion import search_all_sources, process_and_index_paper

# Define state structure
class AgentState(TypedDict):
    query: str
    papers_found: List[Dict[str, Any]]
    downloaded_papers: List[Dict[str, Any]]
    retrieved_chunks: List[Dict[str, Any]]
    answer: str
    citations: Dict[str, Any]
    next_action: str
    errors: List[str]

# Node 1: Check existing knowledge
def check_knowledge_node(state: AgentState) -> AgentState:
    query = state["query"]
    print(f"Checking existing knowledge for: '{query}'")
    
    # Run a quick vector search to see if we have highly relevant local content
    try:
        query_vector = embed_text([query])[0]
        results = search_vectors(query_vector, top_k=3)
        
        # If we have matches with high similarity (> 0.7), we consider it sufficient
        sufficient = False
        if results:
            highest_score = results[0]["score"]
            print(f"Highest similarity score in local index: {highest_score:.4f}")
            if highest_score > 0.7:
                sufficient = True
        else:
            print("No local vector records found.")
            
        # Fallback SQLite check: check if any papers exist at all
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM papers")
        paper_count = cursor.fetchone()["count"]
        conn.close()
        
        # If database is completely empty, we must search web
        if paper_count == 0:
            sufficient = False
            
        if sufficient:
            state["next_action"] = "retrieve"
            print("Local knowledge is SUFFICIENT. Routing to retrieval.")
        else:
            state["next_action"] = "web_search"
            print("Local knowledge is INSUFFICIENT. Routing to web search.")
            
    except Exception as e:
        print(f"Error checking knowledge sufficiency: {e}")
        state["next_action"] = "web_search"
        state["errors"].append(str(e))
        
    return state

# Node 2: Web search
def web_search_node(state: AgentState) -> AgentState:
    query = state["query"]
    print("Initiating Web Search Agent...")
    try:
        papers = search_all_sources(query, limit=5)
        print(f"Found {len(papers)} papers on the web.")
        state["papers_found"] = papers
    except Exception as e:
        print(f"Web search agent error: {e}")
        state["errors"].append(str(e))
        state["papers_found"] = []
    return state

# Node 3: Download and Index papers
def download_and_index_node(state: AgentState) -> AgentState:
    papers = state.get("papers_found", [])
    print("Initiating Download and Index Agent...")
    
    indexed_papers = []
    for paper in papers:
        try:
            print(f"Ingesting paper: {paper['title']}")
            success = process_and_index_paper(paper)
            if success:
                indexed_papers.append(paper)
        except Exception as e:
            print(f"Failed to ingest paper '{paper['title']}': {e}")
            state["errors"].append(str(e))
            
    state["downloaded_papers"] = indexed_papers
    print(f"Ingestion completed. Indexed {len(indexed_papers)}/{len(papers)} papers.")
    return state

# Node 4: Retrieve
def retrieve_node(state: AgentState) -> AgentState:
    query = state["query"]
    print("Initiating Retrieval Agent (Hybrid Search + Reranking)...")
    try:
        chunks = hybrid_search(query)
        print(f"Retrieved {len(chunks)} chunks.")
        state["retrieved_chunks"] = chunks
    except Exception as e:
        print(f"Retrieval error: {e}")
        state["errors"].append(str(e))
        state["retrieved_chunks"] = []
    return state

# Node 5: Generate
def generate_node(state: AgentState) -> AgentState:
    query = state["query"]
    chunks = state.get("retrieved_chunks", [])
    
    if not chunks:
        state["answer"] = "I don't have enough information to answer your question."
        state["citations"] = {}
        return state
        
    print("Initiating Generation Agent...")
    try:
        answer, citations = generate_answer(query, chunks)
        state["answer"] = answer
        state["citations"] = citations
    except Exception as e:
        print(f"Generation error: {e}")
        state["answer"] = "An error occurred while generating the answer. Please try again."
        state["citations"] = {}
        state["errors"].append(str(e))
    return state

# Routing logic function
def route_knowledge(state: AgentState) -> str:
    return state.get("next_action", "web_search")

# Build LangGraph workflow
def build_supervisor_graph():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("check_knowledge", check_knowledge_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("download_and_index", download_and_index_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "check_knowledge",
        route_knowledge,
        {
            "retrieve": "retrieve",
            "web_search": "web_search"
        }
    )
    
    # Add standard sequential edges
    workflow.add_edge(START, "check_knowledge")
    workflow.add_edge("web_search", "download_and_index")
    workflow.add_edge("download_and_index", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()

# Unified run function
def run_query_rag(query: str) -> Dict[str, Any]:
    graph = build_supervisor_graph()
    
    initial_state = {
        "query": query,
        "papers_found": [],
        "downloaded_papers": [],
        "retrieved_chunks": [],
        "answer": "",
        "citations": {},
        "next_action": "",
        "errors": []
    }
    
    final_state = graph.invoke(initial_state)
    return {
        "query": final_state["query"],
        "answer": final_state["answer"],
        "citations": final_state["citations"],
        "papers_found": final_state["papers_found"],
        "downloaded_papers": final_state["downloaded_papers"],
        "retrieved_chunks": final_state["retrieved_chunks"],
        "errors": final_state["errors"]
    }
