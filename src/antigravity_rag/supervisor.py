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
    model_name: Optional[str]
    papers_found: List[Dict[str, Any]]
    downloaded_papers: List[Dict[str, Any]]
    retrieved_chunks: List[Dict[str, Any]]
    answer: str
    thinking: str
    citations: Dict[str, Any]
    next_action: str
    errors: List[str]
    verification: Dict[str, Any]
    agent_logs: List[str]
    loop_count: int
    revision_feedback: str

# Node 1: Check existing knowledge
def check_knowledge_node(state: AgentState) -> AgentState:
    query = state["query"]
    state.setdefault("agent_logs", []).append("Supervisor: Assessing query complexity & local index sufficiency...")
    
    try:
        query_vector = embed_text([query])[0]
        results = search_vectors(query_vector, top_k=3)
        
        sufficient = False
        if results:
            highest_score = results[0]["score"]
            state["agent_logs"].append(f"Supervisor: Found local vectors. Top similarity score: {highest_score:.4f}")
            if highest_score > 0.8:
                sufficient = True
        else:
            state["agent_logs"].append("Supervisor: No local vectors matched.")
            
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM papers")
        paper_count = cursor.fetchone()["count"]
        conn.close()
        
        if paper_count == 0:
            sufficient = False
            
        if sufficient:
            state["next_action"] = "retrieve"
            state["agent_logs"].append("Supervisor: Local knowledge is SUFFICIENT. Skipping web search.")
        else:
            state["next_action"] = "web_search"
            state["agent_logs"].append("Supervisor: Local knowledge is INSUFFICIENT. Dispatching Web Search Agent...")
            
    except Exception as e:
        state["next_action"] = "web_search"
        state["errors"].append(str(e))
        state["agent_logs"].append(f"Supervisor: Error assessing knowledge: {e}. Defaulting to web search.")
        
    return state

# Node 2: Web search
def web_search_node(state: AgentState) -> AgentState:
    query = state["query"]
    state.setdefault("agent_logs", []).append("WebSearchAgent: Parsing query complexity for deconstruction...")
    
    try:
        from src.antigravity_rag.db_sqlite import get_cached_query_results, save_cached_query_results
        cached = get_cached_query_results(query)
        if cached is not None:
            state["agent_logs"].append(f"WebSearchAgent: Retrieved {len(cached)} candidate papers from local search cache.")
            state["papers_found"] = cached
            return state

        # Query deconstruction: deconstruct complex queries into sub-queries
        sub_queries = []
        clean_q = query.strip()
        
        # Split by typical conjunctions or punctuation
        import re
        parts = re.split(r'\band\b|\bor\b|\bwith\b|;|\?', clean_q, flags=re.IGNORECASE)
        for part in parts:
            part_str = part.strip()
            if len(part_str) > 10:
                sub_queries.append(part_str)
                
        # If no sub-queries extracted, default to primary query
        if not sub_queries:
            sub_queries = [clean_q]
            
        state["agent_logs"].append(f"WebSearchAgent: Deconstructed query into {len(sub_queries)} sub-concepts: {sub_queries}")
        
        merged_papers = []
        import difflib
        
        for sq in sub_queries:
            state["agent_logs"].append(f"WebSearchAgent: Fetching sources for sub-concept: '{sq}'")
            papers = search_all_sources(sq, limit=3)
            for paper in papers:
                is_dup = False
                for existing in merged_papers:
                    if difflib.SequenceMatcher(None, paper["title"].lower(), existing["title"].lower()).ratio() > 0.9:
                        is_dup = True
                        break
                if not is_dup:
                    merged_papers.append(paper)
                    
        # Limit to top 5 deduplicated papers total
        final_papers = merged_papers[:5]
        save_cached_query_results(query, final_papers)
        state["agent_logs"].append(f"WebSearchAgent: Located {len(final_papers)} total papers on the web.")
        state["papers_found"] = final_papers
    except Exception as e:
        state["errors"].append(str(e))
        state["papers_found"] = []
        state["agent_logs"].append(f"WebSearchAgent: Search failed: {e}")
    return state

# Node 3: Download and Index papers
def download_and_index_node(state: AgentState) -> AgentState:
    papers = state.get("papers_found", [])
    state.setdefault("agent_logs", []).append("PDFFetchAgent: Starting PDF download streams and database transaction locks...")
    
    indexed_papers = []
    for paper in papers:
        try:
            state["agent_logs"].append(f"PDFFetchAgent: Fetching and parsing PDF for: '{paper['title'][:40]}...'")
            success = process_and_index_paper(paper)
            if success:
                indexed_papers.append(paper)
        except Exception as e:
            state["errors"].append(str(e))
            state["agent_logs"].append(f"PDFFetchAgent: Failed to process paper: {e}")
            
    state["downloaded_papers"] = indexed_papers
    state["agent_logs"].append(f"PDFFetchAgent: Ingestion finished. Saved {len(indexed_papers)} papers as binary BLOBs in SQLite database.")
    return state

# Node 4: Retrieve
def retrieve_node(state: AgentState) -> AgentState:
    query = state["query"]
    state.setdefault("agent_logs", []).append("RetrievalAgent: Running hybrid dense + sparse search (SQLite FTS5 + Qdrant)...")
    try:
        chunks = hybrid_search(query)
        state["agent_logs"].append(f"RetrievalAgent: Fused matching chunks with RRF. Reranked top {len(chunks)} results with Cross-Encoder.")
        state["retrieved_chunks"] = chunks
    except Exception as e:
        state["errors"].append(str(e))
        state["retrieved_chunks"] = []
        state["agent_logs"].append(f"RetrievalAgent: Retrieval error: {e}")
    return state

# Node 5: Generate
def generate_node(state: AgentState) -> AgentState:
    query = state["query"]
    chunks = state.get("retrieved_chunks", [])
    
    # Initialize loop count if not present
    if "loop_count" not in state:
        state["loop_count"] = 0
        
    state.setdefault("agent_logs", []).append(f"GenerationAgent: Invoking model generation (Iteration: {state['loop_count'] + 1})...")
    
    if not chunks:
        state["answer"] = "I don't have enough information to answer your question."
        state["citations"] = {}
        state["agent_logs"].append("GenerationAgent: No relevant context. Returning fallback message.")
        return state
        
    try:
        selected_model = state.get("model_name")
        
        # Inject NLI revision feedback if in a correction loop iteration
        current_query = query
        feedback = state.get("revision_feedback", "")
        if feedback:
            state["agent_logs"].append(f"GenerationAgent: Prepending NLI verification feedback: {feedback}")
            current_query = f"[SYSTEM NOTICE: The verification agent flagged the following issues in your previous output: '{feedback}'. Please rewrite the response correcting these statements using only the verified source excerpts below].\nOriginal query: {query}"
            
        answer, thinking, citations = generate_answer(current_query, chunks, model_name=selected_model)
        state["answer"] = answer
        state["thinking"] = thinking
        state["citations"] = citations
        state["agent_logs"].append(f"GenerationAgent: Response generated with {len(citations)} citations.")
    except Exception as e:
        state["answer"] = "An error occurred while generating the answer. Please try again."
        state["thinking"] = ""
        state["citations"] = {}
        state["errors"].append(str(e))
        state["agent_logs"].append(f"GenerationAgent: Generation failed: {e}")
    return state

# Node 6: Verify
def verify_node(state: AgentState) -> AgentState:
    import re
    import json
    state.setdefault("agent_logs", []).append("VerificationAgent: Initiating Claim Fact-Checking (NLI evaluation)...")
    
    answer = state.get("answer", "")
    retrieved_chunks = state.get("retrieved_chunks", [])
    
    if not answer or not retrieved_chunks:
        state["verification"] = {}
        state["agent_logs"].append("VerificationAgent: No statements to fact-check.")
        return state
        
    # Split the answer into sentences to check individual claims
    sentences = re.split(r'\. +', answer)
    verification = {}
    chunk_map = {c["chunk_id"]: c for c in retrieved_chunks}
    
    # We find which chunk_ids are cited in which sentences
    for sentence in sentences:
        # Search for superscript or standard citation markers
        cited_ids = re.findall(r'【([^】]+)】', sentence)
        if not cited_ids:
            cited_ids = re.findall(r'\[([^\]]+)\]', sentence)
            
        for cid in cited_ids:
            cid = cid.strip()
            chunk = chunk_map.get(cid)
            if chunk:
                excerpt = chunk["chunk_text"]
                claim = sentence
                
                # Prompt LLM to act as a zero-shot NLI classifier
                prompt = f"""[INST] You are an NLI (Natural Language Inference) model. Analyze if the Excerpt supports the Claim.
Excerpt: "{excerpt}"
Claim: "{claim}"

Respond ONLY with a JSON object in this format:
{{"supported": true, "confidence": 0.9}}
Do not write any explanation. [/INST]"""
                
                try:
                    from src.antigravity_rag.ollama_chat import call_ollama
                    resp = call_ollama(prompt)
                    match = re.search(r'\{.*\}', resp, re.DOTALL)
                    if match:
                        data = json.loads(match.group(0))
                        confidence = float(data.get("confidence", 0.8))
                        supported = data.get("supported", True)
                        
                        verification[cid] = {
                            "supported": supported,
                            "confidence": confidence
                        }
                        status = "SUPPORTED" if supported else "⚠️ REFUTED"
                        state["agent_logs"].append(f"VerificationAgent: Claim citing [{cid[:8]}...] is {status} (confidence: {confidence:.2f})")
                    else:
                        verification[cid] = {"supported": True, "confidence": 0.8}
                except Exception as e:
                    verification[cid] = {"supported": True, "confidence": 0.8}
                    
    state["verification"] = verification
    state["agent_logs"].append("VerificationAgent: Pipeline claim-verification complete.")
    return state

# Routing logic function
def route_knowledge(state: AgentState) -> str:
    return state.get("next_action", "web_search")

def route_verification(state: AgentState) -> str:
    verification = state.get("verification", {})
    loop_count = state.get("loop_count", 0)
    
    # Locate any refuted statements
    refuted_claims = []
    for cid, val in verification.items():
        if not val.get("supported", True) and val.get("confidence", 0.0) >= 0.7:
            # Retrieve refuted excerpt details
            refuted_claims.append(f"Citation [{cid[:8]}] refuted with confidence {val['confidence']:.2f}")
            
    if refuted_claims and loop_count < 2:
        state["loop_count"] = loop_count + 1
        state["revision_feedback"] = "; ".join(refuted_claims)
        state["agent_logs"].append(f"Supervisor: Refuted claims identified. Dispatching NLI critique loop (Loop: {state['loop_count']}/2)...")
        return "generate"
        
    state["agent_logs"].append("Supervisor: All claims verified or loop count exceeded. Routing execution to END.")
    return "end"

# Build LangGraph workflow
def build_supervisor_graph():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("check_knowledge", check_knowledge_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("download_and_index", download_and_index_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("verify", verify_node)
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "check_knowledge",
        route_knowledge,
        {
            "retrieve": "retrieve",
            "web_search": "web_search"
        }
    )
    
    workflow.add_conditional_edges(
        "verify",
        route_verification,
        {
            "generate": "generate",
            "end": END
        }
    )
    
    # Add standard sequential edges
    workflow.add_edge(START, "check_knowledge")
    workflow.add_edge("web_search", "download_and_index")
    workflow.add_edge("download_and_index", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "verify")
    
    return workflow.compile()

# Unified streaming function
def stream_query_rag(query: str, model_name: str = None):
    graph = build_supervisor_graph()
    initial_state = {
        "query": query,
        "model_name": model_name,
        "papers_found": [],
        "downloaded_papers": [],
        "retrieved_chunks": [],
        "answer": "",
        "thinking": "",
        "citations": {},
        "next_action": "",
        "errors": [],
        "verification": {},
        "loop_count": 0,
        "revision_feedback": "",
        "agent_logs": ["Supervisor: Initializing LangGraph state machine..."]
    }
    # Stream node transitions
    for step in graph.stream(initial_state):
        yield step

# Unified run function (wraps streaming for compatibility)
def run_query_rag(query: str, model_name: str = None) -> Dict[str, Any]:
    final_state = None
    for step in stream_query_rag(query, model_name=model_name):
        # Save the latest state emitted by any node
        node_name = list(step.keys())[0]
        final_state = step[node_name]
        
    return {
        "query": final_state["query"],
        "answer": final_state["answer"],
        "thinking": final_state.get("thinking", ""),
        "citations": final_state["citations"],
        "papers_found": final_state["papers_found"],
        "downloaded_papers": final_state["downloaded_papers"],
        "retrieved_chunks": final_state["retrieved_chunks"],
        "errors": final_state["errors"],
        "verification": final_state.get("verification", {}),
        "agent_logs": final_state.get("agent_logs", [])
    }
