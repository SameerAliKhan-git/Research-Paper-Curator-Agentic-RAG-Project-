import logging
from typing import List
from datetime import datetime, timedelta
import os

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy import select, delete

from src.dependencies import SessionDep, EmbeddingsDep, OllamaDep
from src.models.researcher import ResearcherInterest, DailyBriefing
from src.schemas.api.recommendations import InterestCreate, InterestResponse, BriefingResponse, RecommendationTriggerRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/recommendations", tags=["recommendations"])

@router.post("/interests", response_model=InterestResponse)
async def add_interest(request: InterestCreate, db: SessionDep) -> InterestResponse:
    """Add a research interest keyword to rank preprints."""
    # Check if keyword already exists
    existing = db.scalars(select(ResearcherInterest).where(ResearcherInterest.keyword == request.keyword)).first()
    if existing:
        raise HTTPException(status_code=400, detail="Interest keyword already exists")
    
    interest = ResearcherInterest(keyword=request.keyword)
    db.add(interest)
    db.commit()
    db.refresh(interest)
    return interest

@router.get("/interests", response_model=List[InterestResponse])
async def list_interests(db: SessionDep) -> List[InterestResponse]:
    """List all current research interest keywords."""
    interests = db.scalars(select(ResearcherInterest).order_by(ResearcherInterest.keyword)).all()
    return list(interests)

@router.delete("/interests/{keyword}")
async def delete_interest(keyword: str, db: SessionDep):
    """Delete a research interest keyword."""
    interest = db.scalars(select(ResearcherInterest).where(ResearcherInterest.keyword == keyword)).first()
    if not interest:
        raise HTTPException(status_code=404, detail="Interest keyword not found")
    
    db.delete(interest)
    db.commit()
    return {"status": "success", "message": f"Deleted interest: {keyword}"}

@router.get("/briefings", response_model=List[BriefingResponse])
async def get_briefings(db: SessionDep, limit: int = 10) -> List[BriefingResponse]:
    """Get high-scoring daily preprints briefings."""
    briefings = db.scalars(
        select(DailyBriefing)
        .order_by(DailyBriefing.score.desc(), DailyBriefing.created_at.desc())
        .limit(limit)
    ).all()
    return list(briefings)

@router.post("/generate")
async def generate_briefings(
    request: RecommendationTriggerRequest,
    db: SessionDep,
    embeddings_service: EmbeddingsDep,
    ollama_client: OllamaDep,
):
    """Manually trigger daily briefings generation for a target date."""
    target_date = request.target_date
    if not target_date:
        target_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")

    logger.info(f"Manually triggering daily briefings for target date: {target_date}")
    
    # Imports for briefing generation
    from src.services.arxiv.factory import make_arxiv_client
    arxiv_client = make_arxiv_client()
    
    interests = db.scalars(select(ResearcherInterest)).all()
    keywords = [i.keyword for i in interests]
    
    if not keywords:
        keywords = ["AI", "RAG", "LLM", "Agentic RAG", "Retrieval-Augmented Generation", "Reinforcement Learning"]
        
    search_query = f"(cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:cs.CV) AND submittedDate:[{target_date}0000 TO {target_date}2359]"
    
    try:
        papers = await arxiv_client.fetch_papers_with_query(search_query, max_results=20)
    except Exception as e:
        logger.error(f"Failed to fetch papers from arXiv: {e}")
        raise HTTPException(status_code=500, detail=f"ArXiv query failed: {e}")
        
    if not papers:
        # Fallback query
        fallback_query = "cat:cs.AI OR cat:cs.LG OR cat:cs.CL"
        try:
            papers = await arxiv_client.fetch_papers_with_query(fallback_query, max_results=10)
        except Exception as e:
            logger.error(f"Fallback arXiv fetch failed: {e}")
            raise HTTPException(status_code=500, detail=f"Fallback ArXiv query failed: {e}")

    if not papers:
        return {"status": "skipped", "message": "No new papers found on arXiv to process."}

    # Rank papers by calculating similarity to interest keywords
    try:
        # Embed keywords
        keyword_vecs = []
        for kw in keywords:
            vec = await embeddings_service.embed_query(kw)
            keyword_vecs.append(vec)
            
        # Embed paper abstracts
        paper_texts = [f"{p.title}: {p.abstract}" for p in papers]
        paper_vecs = await embeddings_service.embed_passages(paper_texts)
        
        # Calculate cosine similarity
        import math
        def cosine_similarity(v1, v2):
            dot_prod = sum(a * b for a, b in zip(v1, v2))
            mag1 = math.sqrt(sum(a * a for a in v1))
            mag2 = math.sqrt(sum(b * b for b in v2))
            if mag1 == 0 or mag2 == 0:
                return 0.0
            return dot_prod / (mag1 * mag2)
            
        scored_papers = []
        for i, paper in enumerate(papers):
            paper_vec = paper_vecs[i]
            scores = [cosine_similarity(paper_vec, kw_vec) for kw_vec in keyword_vecs]
            max_score = max(scores) if scores else 0.0
            scored_papers.append((max_score, paper))
            
        scored_papers.sort(key=lambda x: x[0], reverse=True)
        top_scored = scored_papers[:5]
    except Exception as e:
        logger.error(f"Embedding or scoring failed: {e}")
        top_scored = [(1.0, p) for p in papers[:5]]

    briefings_inserted = 0
    for score, paper in top_scored:
        # Check if already exists
        existing = db.scalars(select(DailyBriefing).where(DailyBriefing.arxiv_id == paper.arxiv_id)).first()
        if existing:
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
            model_name = os.environ.get("OLLAMA_MODEL", "gemma3:latest")
            resp = await ollama_client.generate(model=model_name, prompt=prompt)
            if resp and "response" in resp:
                summary = resp["response"].strip()
        except Exception as e:
            logger.error(f"Ollama summary generation failed for {paper.arxiv_id}: {e}")
            
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
        db.add(briefing)
        briefings_inserted += 1
        
    db.commit()
    return {"status": "success", "briefings_inserted": briefings_inserted}
