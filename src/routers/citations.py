import logging

from fastapi import APIRouter, HTTPException
from src.dependencies import APIKeyDep, SessionDep
from src.repositories.paper import PaperRepository
from src.schemas.api import validate_arxiv_id
from src.schemas.api.citations import CitationEdge, CitationGraph, CitationNode

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/papers", tags=["citations"])


@router.get("/{arxiv_id}/citations", response_model=CitationGraph)
async def get_citations(arxiv_id: str, db: SessionDep, _key: APIKeyDep) -> CitationGraph:
    """Return a citation graph for the given paper."""
    arxiv_id = validate_arxiv_id(arxiv_id)
    repo = PaperRepository(db)
    paper = repo.get_by_arxiv_id(arxiv_id)

    if not paper:
        raise HTTPException(status_code=404, detail=f"Paper {arxiv_id} not found")

    nodes: list[CitationNode] = []
    edges: list[CitationEdge] = []
    seen_ids: set[str] = set()

    nodes.append(CitationNode(id=paper.arxiv_id, title=paper.title))
    seen_ids.add(paper.arxiv_id)

    references = paper.references if isinstance(paper.references, list) else []
    for ref_id in references:
        if not isinstance(ref_id, str) or not ref_id.strip():
            continue
        ref_id = ref_id.strip()
        edges.append(CitationEdge(source=paper.arxiv_id, target=ref_id))
        if ref_id not in seen_ids:
            ref_paper = repo.get_by_arxiv_id(ref_id)
            title = ref_paper.title if ref_paper else ref_id
            nodes.append(CitationNode(id=ref_id, title=title))
            seen_ids.add(ref_id)

    citing_papers = repo.get_papers_citing(arxiv_id)
    for citing in citing_papers:
        edges.append(CitationEdge(source=citing.arxiv_id, target=paper.arxiv_id))
        if citing.arxiv_id not in seen_ids:
            nodes.append(CitationNode(id=citing.arxiv_id, title=citing.title))
            seen_ids.add(citing.arxiv_id)

    return CitationGraph(
        paper_id=paper.arxiv_id,
        nodes=nodes,
        edges=edges,
    )
