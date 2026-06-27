import logging
from typing import List

from fastapi import APIRouter, HTTPException
from src.dependencies import OllamaDep, OpenSearchDep, SessionDep
from src.repositories.paper import PaperRepository
from src.schemas.api.summarize import SummarizeRequest, SummarizeResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/papers", tags=["summarize"])


def _build_summarization_prompt(text: str, summary_type: str) -> str:
    """Build a summarization prompt based on the summary type."""
    if summary_type == "brief":
        return (
            "Provide a concise 2-3 paragraph summary of the following research paper. "
            "Focus on the main contribution and key results.\n\n"
            f"PAPER TEXT:\n{text}"
        )
    if summary_type == "detailed":
        return (
            "Provide a detailed summary of the following research paper, covering: "
            "1) Background and motivation, 2) Methodology, 3) Key results and findings, "
            "4) Limitations and future work. Use 4-6 paragraphs.\n\n"
            f"PAPER TEXT:\n{text}"
        )
    # technical
    return (
        "Provide a technical summary of the following research paper. Include: "
        "1) Problem formulation, 2) Mathematical approach and model architecture, "
        "3) Training procedure, 4) Quantitative results with specific metrics, "
        "5) Ablation studies if present. Be precise with numbers and technical terms.\n\n"
        f"PAPER TEXT:\n{text}"
    )


def _build_key_findings_prompt(text: str) -> str:
    """Build a prompt to extract key findings."""
    return (
        "Extract the 3-5 most important findings or contributions from the following "
        "research paper. Return ONLY a numbered list, one finding per line, with no "
        "additional explanation.\n\n"
        f"PAPER TEXT:\n{text}"
    )


def _parse_findings(raw: str) -> List[str]:
    """Parse a numbered/bulleted list of findings from LLM output."""
    findings: List[str] = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        for prefix in ("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "-", "•"):
            if line.startswith(prefix):
                line = line[len(prefix) :].strip()
                break
        if line:
            findings.append(line)
    return findings[:5]


async def _get_paper_text(
    paper: object,
    open_search_client,
) -> str:
    """Return the paper's full text, preferring raw_text from PostgreSQL and falling back to OpenSearch chunks."""
    raw_text = getattr(paper, "raw_text", None)
    if raw_text:
        return raw_text

    chunks = open_search_client.get_chunks_by_paper(paper.arxiv_id)
    if chunks:
        return "\n\n".join(c.get("chunk_text", "") for c in chunks)

    return paper.abstract or ""


@router.post("/{arxiv_id}/summarize", response_model=SummarizeResponse)
async def summarize_paper(
    arxiv_id: str,
    request: SummarizeRequest,
    db: SessionDep,
    ollama_client: OllamaDep,
    open_search_client: OpenSearchDep,
) -> SummarizeResponse:
    """Summarize a research paper using an LLM."""
    repo = PaperRepository(db)
    paper = repo.get_by_arxiv_id(arxiv_id)

    if not paper:
        raise HTTPException(status_code=404, detail=f"Paper {arxiv_id} not found")

    text = await _get_paper_text(paper, open_search_client)
    if not text:
        raise HTTPException(status_code=422, detail="Paper has no usable text content for summarization")

    prompt = _build_summarization_prompt(text, request.summary_type)

    try:
        response = await ollama_client.generate(
            model=request.model,
            prompt=prompt,
            temperature=0.4,
            top_p=0.9,
        )
    except Exception as e:
        logger.error(f"Ollama generation failed for {arxiv_id}: {e}")
        raise HTTPException(status_code=500, detail="LLM generation failed")

    if not response or "response" not in response:
        raise HTTPException(status_code=500, detail="Empty response from LLM")

    summary_text = response["response"]

    findings_prompt = _build_key_findings_prompt(text)
    try:
        findings_response = await ollama_client.generate(
            model=request.model,
            prompt=findings_prompt,
            temperature=0.2,
            top_p=0.9,
        )
        key_findings = _parse_findings(findings_response.get("response", ""))
    except Exception as e:
        logger.warning(f"Key findings extraction failed for {arxiv_id}: {e}")
        key_findings = []

    categories = paper.categories if isinstance(paper.categories, list) else [paper.categories] if paper.categories else []

    return SummarizeResponse(
        arxiv_id=paper.arxiv_id,
        title=paper.title,
        summary=summary_text,
        key_findings=key_findings,
        categories=categories,
    )
