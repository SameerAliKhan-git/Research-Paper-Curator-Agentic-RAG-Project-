import logging
from typing import List

from fastapi import APIRouter, HTTPException
from src.dependencies import EmbeddingsDep, OllamaDep, OpenSearchDep, SessionDep
from src.repositories.paper import PaperRepository
from src.schemas.api.review import ReviewPaper, ReviewRequest, ReviewResponse, ReviewSection

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/papers", tags=["review"])

REVIEW_SYSTEM_PROMPT = (
    "You are a research assistant. Write a structured literature review section "
    "based on the provided papers. For each section, provide a heading and a "
    "well-written synthesis of the findings. Use academic writing style."
)


def _extract_findings(abstract: str) -> List[str]:
    """Extract key findings from an abstract by sentence splitting."""
    sentences = [s.strip() for s in abstract.replace("\n", " ").split(".") if len(s.strip()) > 30]
    return sentences[:3]


async def _synthesize_review(
    ollama_client,
    topic: str,
    papers: List[ReviewPaper],
    model: str,
) -> List[ReviewSection]:
    """Use LLM to synthesize a literature review from retrieved papers."""
    paper_summaries = []
    for p in papers:
        findings = "; ".join(p.key_findings) if p.key_findings else "N/A"
        paper_summaries.append(f"- {p.title} ({p.arxiv_id}): {findings}")

    papers_text = "\n".join(paper_summaries)
    prompt = (
        f"{REVIEW_SYSTEM_PROMPT}\n\n"
        f"Topic: {topic}\n\n"
        f"Papers:\n{papers_text}\n\n"
        "Write the review with the following sections:\n"
        "1. Overview - introduction to the research area\n"
        "2. Key Approaches - main methods and architectures discussed\n"
        "3. Comparative Analysis - strengths and weaknesses across papers\n"
        "4. Open Questions - gaps and future research directions\n\n"
        "Return each section as:\nSECTION: <heading>\n<content>\n"
    )

    try:
        response = await ollama_client.generate(
            model=model,
            prompt=prompt,
            temperature=0.5,
            top_p=0.9,
        )
    except Exception as e:
        logger.error(f"LLM generation failed for literature review: {e}")
        raise HTTPException(status_code=500, detail="LLM generation failed")

    if not response or "response" not in response:
        raise HTTPException(status_code=500, detail="Empty response from LLM")

    raw = response["response"]
    sections: List[ReviewSection] = []
    current_heading = ""
    current_lines: list[str] = []

    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("SECTION:"):
            if current_heading and current_lines:
                sections.append(ReviewSection(heading=current_heading, content="\n".join(current_lines).strip()))
            current_heading = stripped[len("SECTION:") :].strip()
            current_lines = []
        else:
            if stripped:
                current_lines.append(stripped)

    if current_heading and current_lines:
        sections.append(ReviewSection(heading=current_heading, content="\n".join(current_lines).strip()))

    if not sections:
        sections = [ReviewSection(heading="Review", content=raw.strip())]

    return sections


@router.post("/literature-review", response_model=ReviewResponse)
async def literature_review(
    request: ReviewRequest,
    db: SessionDep,
    opensearch_client: OpenSearchDep,
    embeddings_service: EmbeddingsDep,
    ollama_client: OllamaDep,
) -> ReviewResponse:
    """Generate a literature review for a given topic."""
    try:
        query_embedding_response = await embeddings_service.embed(request.topic)
        query_embedding = query_embedding_response[0] if query_embedding_response else None
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail="Embedding generation failed")

    categories = [request.category] if request.category else None

    try:
        results = await opensearch_client.search_unified(
            query=request.topic,
            query_embedding=query_embedding,
            size=10,
            categories=categories,
            use_hybrid=query_embedding is not None,
        )
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail="Search failed")

    hits = results.get("hits", [])
    if not hits:
        raise HTTPException(status_code=404, detail="No relevant papers found for the topic")

    repo = PaperRepository(db)
    review_papers: List[ReviewPaper] = []
    seen_ids: set[str] = set()

    for hit in hits:
        arxiv_id = hit.get("arxiv_id", "")
        if not arxiv_id or arxiv_id in seen_ids:
            continue
        seen_ids.add(arxiv_id)

        paper = repo.get_by_arxiv_id(arxiv_id)
        abstract = paper.abstract if paper else hit.get("abstract", "")
        findings = _extract_findings(abstract)

        review_papers.append(
            ReviewPaper(
                arxiv_id=arxiv_id,
                title=paper.title if paper else hit.get("title", arxiv_id),
                key_findings=findings,
            )
        )

    sections = await _synthesize_review(
        ollama_client=ollama_client,
        topic=request.topic,
        papers=review_papers,
        model=request.model,
    )

    return ReviewResponse(
        topic=request.topic,
        sections=sections,
        papers=review_papers,
    )
