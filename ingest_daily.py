import sys
import argparse
import time
from datetime import datetime, timedelta
from src.antigravity_rag.config_parser import get_config
from src.antigravity_rag.db_sqlite import init_db
from src.antigravity_rag.db_qdrant import init_qdrant
from src.antigravity_rag.ingestion import search_arxiv, search_semantic_scholar, process_and_index_paper

# Safe apscheduler import
try:
    from apscheduler.schedulers.blocking import BlockingScheduler
    APSCHEDULER_AVAILABLE = True
except ImportError:
    APSCHEDULER_AVAILABLE = False

def run_ingestion():
    print(f"[{datetime.now().isoformat()}] Starting scheduled ingestion process...")
    config = get_config()
    
    # Initialize DBs
    init_db()
    init_qdrant()
    
    sources_cfg = config.sources
    arxiv_cfg = sources_cfg.get("arxiv", {})
    ss_cfg = sources_cfg.get("semantic_scholar", {})
    
    all_papers = []
    
    # 1. Fetch from ArXiv
    if arxiv_cfg.get("enabled", True):
        categories = arxiv_cfg.get("categories", ["cs.AI", "cs.CL", "cs.LG"])
        max_results = arxiv_cfg.get("max_results", 10)
        
        # Build query for recent papers in the configured categories
        query_parts = [f"cat:{cat}" for cat in categories]
        query = " OR ".join(query_parts)
        print(f"Fetching from ArXiv with query: {query}")
        
        # Search recent papers
        recent_arxiv = search_arxiv(query, max_results=max_results)
        all_papers.extend(recent_arxiv)
        
    # 2. Fetch from Semantic Scholar
    if ss_cfg.get("enabled", True):
        max_results = ss_cfg.get("max_results", 5)
        # Search using general hot categories or keywords
        keywords = ["large language models", "retrieval augmented generation", "agentic workflows"]
        for kw in keywords[:2]:
            print(f"Fetching from Semantic Scholar for keyword: '{kw}'")
            recent_ss = search_semantic_scholar(kw, max_results=max_results)
            all_papers.extend(recent_ss)
            
    # Deduplicate results by title
    unique_papers = []
    seen_titles = set()
    for paper in all_papers:
        title_lower = paper["title"].lower().strip()
        if title_lower not in seen_titles:
            seen_titles.add(title_lower)
            unique_papers.append(paper)
            
    print(f"Found total of {len(unique_papers)} unique papers to process.")
    
    success_count = 0
    for idx, paper in enumerate(unique_papers):
        print(f"[{idx+1}/{len(unique_papers)}] Ingesting paper: {paper['title']}")
        try:
            success = process_and_index_paper(paper)
            if success:
                success_count += 1
        except Exception as e:
            print(f"Failed to ingest paper: {e}")
            
    print(f"Daily ingestion completed. Successfully processed {success_count}/{len(unique_papers)} new papers.")

def start_daemon():
    if not APSCHEDULER_AVAILABLE:
        print("apscheduler is not installed. Cannot run in daemon mode. Install it or run without --daemon.")
        sys.exit(1)
        
    config = get_config()
    schedule_cfg = config.schedule
    hour = schedule_cfg.get("daily_ingestion_hour", 3)
    timezone = schedule_cfg.get("timezone", "UTC")
    
    scheduler = BlockingScheduler(timezone=timezone)
    # Trigger at daily_ingestion_hour:00
    scheduler.add_job(run_ingestion, 'cron', hour=hour, minute=0)
    
    print(f"Daemon started. Scheduled daily ingestion at {hour:02d}:00 {timezone}.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("Daemon stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project Antigravity Daily Ingestion Service")
    parser.add_argument("--daemon", action="store_true", help="Run as background daemon scheduled daily")
    parser.add_argument("--now", action="store_true", help="Run ingestion immediately and exit")
    
    args = parser.parse_args()
    
    if args.daemon:
        start_daemon()
    else:
        # Run immediately by default
        run_ingestion()
