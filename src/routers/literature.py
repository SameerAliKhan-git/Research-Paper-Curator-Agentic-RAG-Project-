import io
import json
import logging
import zipfile
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from src.dependencies import APIKeyDep, OllamaDep, SessionDep
from src.repositories.paper import PaperRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/literature", tags=["literature"])


class RelatedWorkRequest(BaseModel):
    arxiv_ids: List[str]
    model: Optional[str] = "llama3.2:1b"


@router.post("/related-work")
async def generate_related_work(
    request: RelatedWorkRequest,
    db: SessionDep,
    ollama_client: OllamaDep,
    _key: APIKeyDep = None,
):
    """
    Generate a LaTeX comparative literature review ("Related Work") section
    and a matching BibTeX bibliography file (.bib) for selected papers.
    """
    if not request.arxiv_ids:
        raise HTTPException(status_code=400, detail="Must provide at least one arXiv ID")

    repo = PaperRepository(db)
    papers = []

    # Retrieve papers from database
    for arxiv_id in request.arxiv_ids:
        paper = repo.get_by_arxiv_id(arxiv_id)
        if paper:
            papers.append(paper)

    if not papers:
        raise HTTPException(
            status_code=404, detail="None of the requested papers were found in the database. Please ingest them first."
        )

    # Compile prompt data for Ollama
    paper_context = []
    bibtex_entries = []

    for paper in papers:
        # 1. Format metadata context for prompt
        paper_context.append(f"Paper ID: {paper.arxiv_id}\nTitle: {paper.title}\nAbstract: {paper.abstract}\n")

        # 2. Build BibTeX string
        authors_val = paper.authors
        if isinstance(authors_val, str):
            try:
                parsed = json.loads(authors_val)
                authors_list = parsed if isinstance(parsed, list) else [str(parsed)]
            except Exception:
                authors_list = [authors_val]
        elif isinstance(authors_val, list):
            authors_list = authors_val
        else:
            authors_list = []

        authors_str = " and ".join(authors_list) if authors_list else "Unknown"
        year = paper.published_date.year if paper.published_date else 2026
        bib_key = f"arxiv_{paper.arxiv_id.replace('.', '_').replace('/', '_')}"

        bib_entry = f"""@article{{{bib_key},
  author    = {{{authors_str}}},
  title     = {{{paper.title}}},
  journal   = {{arXiv preprint arXiv:{paper.arxiv_id}}},
  year      = {{{year}}},
  url       = {{{paper.pdf_url or f"https://arxiv.org/abs/{paper.arxiv_id}"}}}
}}"""
        bibtex_entries.append(bib_entry)

    papers_meta = "\n---\n".join(paper_context)
    bibtex_content = "\n\n".join(bibtex_entries)

    # Construct the synthesis prompt
    prompt = f"""You are a senior AI/ML researcher drafting a "Related Work" section for an academic paper.
Compare and synthesize the contributions of the following retrieved research papers.

Retrieved Papers:
{papers_meta}

Instructions:
1. Write a cohesive, technical comparative literature review of these papers.
2. Group them semantically by topic if appropriate.
3. Cite them using standard LaTeX cite syntax: \\cite{{arxiv_<arxiv_id_with_underscores>}}.
   For example, for paper 1706.03762, the citation key is \\cite{{arxiv_1706_03762}}.
4. Return ONLY valid LaTeX markup. Do not wrap in markdown code blocks. Do not add intro/outro comments.
   Start directly with the LaTeX section title, e.g., \\section{{Related Work}} or \\subsection{{...}}.
"""

    try:
        # Generate the LaTeX section
        logger.info(f"Generating LaTeX related work section using model: {request.model}")
        response = await ollama_client.generate_rag_answer(
            query="Draft LaTeX related work", chunks=[{"chunk_text": prompt, "arxiv_id": "prompt"}], model=request.model
        )
        latex_content = response.get("answer", "").strip()

        # Clean any markdown packaging (e.g. ```latex ... ```) if present
        if latex_content.startswith("```"):
            lines = latex_content.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            latex_content = "\n".join(lines).strip()

    except Exception as e:
        logger.error(f"Failed to generate related work LaTeX: {e}")
        raise HTTPException(status_code=500, detail="LaTeX synthesis failed")

    # Package as an in-memory zip file
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        zip_file.writestr("related_work.tex", latex_content)
        zip_file.writestr("references.bib", bibtex_content)

    zip_buffer.seek(0)

    return StreamingResponse(
        zip_buffer, media_type="application/zip", headers={"Content-Disposition": "attachment; filename=related_work_latex.zip"}
    )
