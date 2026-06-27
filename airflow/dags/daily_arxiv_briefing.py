import sys
import os
sys.path.insert(0, "/opt/airflow")

from datetime import datetime, timedelta
import asyncio
import logging
import math
from airflow import DAG
from airflow.operators.python import PythonOperator

from src.db.factory import make_database
from src.services.arxiv.factory import make_arxiv_client
from src.services.embeddings.factory import make_embeddings_client
from src.services.ollama.factory import make_ollama_client
from src.models.researcher import ResearcherInterest, DailyBriefing

logger = logging.getLogger(__name__)

default_args = {
    "owner": "arxiv-curator",
    "depends_on_past": False,
    "start_date": datetime(2025, 8, 8),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "catchup": False,
}

dag = DAG(
    "daily_arxiv_briefing",
    default_args=default_args,
    description="Daily schedule to fetch new preprints, rank by interest profiles, summarize via Ollama, and save briefings.",
    schedule="0 7 * * *",  # Run daily at 7 AM UTC
    max_active_runs=1,
    catchup=False,
    tags=["arxiv", "briefing", "researcher"],
)


def cosine_similarity(v1, v2):
    dot_prod = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(b * b for b in v2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot_prod / (mag1 * mag2)


async def run_daily_briefing(target_date: str):
    logger.info(f"Starting daily briefing generation for target date: {target_date}")
    
    database = make_database()
    arxiv_client = make_arxiv_client()
    embeddings_client = make_embeddings_client()
    ollama_client = make_ollama_client()
    
    keywords = []
    
    # 1. Fetch user interest keywords from database
    with database.get_session() as session:
        interests = session.query(ResearcherInterest).all()
        keywords = [i.keyword for i in interests]
        
    if not keywords:
        logger.info("No researcher interests found in database. Using default keywords.")
        keywords = ["AI", "RAG", "LLM", "Agentic RAG", "Retrieval-Augmented Generation", "Reinforcement Learning"]
        
    logger.info(f"Using researcher interest keywords: {keywords}")
    
    # 2. Query arXiv API for CS preprints
    # Format dates to arXiv format: YYYYMMDD
    date_from = target_date
    date_to = target_date
    
    # Build query
    search_query = f"(cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:cs.CV) AND submittedDate:[{date_from}0000 TO {date_to}2359]"
    logger.info(f"arXiv query: {search_query}")
    
    papers = []
    try:
        papers = await arxiv_client.fetch_papers_with_query(search_query, max_results=50)
    except Exception as e:
        logger.error(f"Failed to fetch papers with date constraint: {e}")
        
    if not papers:
        logger.info("No papers found for target date. Falling back to fetching the latest 20 preprints from CS.AI/CS.LG/CS.CL.")
        fallback_query = "cat:cs.AI OR cat:cs.LG OR cat:cs.CL"
        try:
            papers = await arxiv_client.fetch_papers_with_query(fallback_query, max_results=20)
        except Exception as e:
            logger.error(f"Fallback arXiv fetch failed: {e}")
            
    if not papers:
        logger.warning("No papers fetched from arXiv. Aborting briefing generation.")
        return {"status": "skipped", "message": "No papers fetched"}

    logger.info(f"Processing {len(papers)} papers for similarity scoring.")
    
    # 3. Calculate embeddings and compute similarity scores
    try:
        # Embed keywords
        keyword_vecs = []
        for kw in keywords:
            vec = await embeddings_client.embed_query(kw)
            keyword_vecs.append(vec)
            
        # Embed paper abstracts
        paper_texts = [f"{p.title}: {p.abstract}" for p in papers]
        paper_vecs = await embeddings_client.embed_passages(paper_texts)
        
        # Calculate max similarity score for each paper
        scored_papers = []
        for i, paper in enumerate(papers):
            paper_vec = paper_vecs[i]
            scores = [cosine_similarity(paper_vec, kw_vec) for kw_vec in keyword_vecs]
            max_score = max(scores) if scores else 0.0
            scored_papers.append((max_score, paper))
            
        # Sort by similarity score descending and take top 5
        scored_papers.sort(key=lambda x: x[0], reverse=True)
        top_scored = scored_papers[:5]
        
    except Exception as e:
        logger.error(f"Embedding or scoring failed: {e}. Defaulting to first 5 papers with score 1.0.")
        top_scored = [(1.0, p) for p in papers[:5]]

    logger.info(f"Top {len(top_scored)} papers selected for daily briefings.")
    
    # 4. Generate summaries using Ollama and insert briefings into database
    briefings_inserted = 0
    with database.get_session() as session:
        for score, paper in top_scored:
            # Check if this arxiv_id already exists in daily_briefings to avoid duplicates
            existing = session.query(DailyBriefing).filter_by(arxiv_id=paper.arxiv_id).first()
            if existing:
                logger.info(f"Briefing for paper {paper.arxiv_id} already exists. Skipping.")
                continue
                
            prompt = (
                "You are a professional research assistant. Summarize the following scientific paper in 3 concise, "
                "highly technical bullet points highlighting its core contribution, methodology, and key results. "
                "Do not include any greeting, intro, or wrap-up. Output ONLY the 3 bullet points.\n\n"
                f"Title: {paper.title}\n"
                f"Abstract: {paper.abstract}\n\n"
                "Summary:"
            )
            
            summary = "Summary generation failed."
            try:
                # Use default fallback model in settings
                model_name = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")
                resp = await ollama_client.generate(model=model_name, prompt=prompt)
                if resp and "response" in resp:
                    summary = resp["response"].strip()
            except Exception as e:
                logger.error(f"Ollama summary generation failed for {paper.arxiv_id}: {e}")
                
            # Parse published date
            try:
                pub_date = datetime.fromisoformat(paper.published_date.replace("Z", "+00:00"))
            except Exception:
                pub_date = datetime.now()
                
            briefing = DailyBriefing(
                arxiv_id=paper.arxiv_id,
                title=paper.title,
                summary=summary,
                score=float(score),
                published_date=pub_date,
            )
            session.add(briefing)
            briefings_inserted += 1
            
        session.commit()
        
    logger.info(f"Daily briefing generation completed. Inserted {briefings_inserted} new briefings.")
    return {"status": "success", "briefings_inserted": briefings_inserted}


def generate_briefing_task(**context):
    execution_date = context.get("execution_date")
    if execution_date:
        target_date = (execution_date - timedelta(days=1)).strftime("%Y%m%d")
    else:
        target_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
        
    return asyncio.run(run_daily_briefing(target_date))


generate_briefing = PythonOperator(
    task_id="generate_briefing",
    python_callable=generate_briefing_task,
    dag=dag,
)
