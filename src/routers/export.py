import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse
from src.dependencies import APIKeyDep, SessionDep
from src.repositories.paper import PaperRepository
from src.schemas.api import validate_arxiv_id

logger = logging.getLogger(__name__)

router = APIRouter(tags=["export"])


def _escape_bibtex(text: str) -> str:
    """Escape special BibTeX characters."""
    if not text:
        return ""
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "{": r"\{",
        "}": r"\}",
    }
    for char, escaped in replacements.items():
        text = text.replace(char, escaped)
    return text


def _format_bibtex(paper: Any) -> str:
    """Format a paper as a BibTeX @article entry.

    Generates a cite key from first author surname + year + arxiv_id suffix.
    """
    authors = paper.authors if isinstance(paper.authors, list) else [paper.authors] if paper.authors else ["Unknown"]
    bibtex_authors = " and ".join(authors)

    first_author_last = authors[0].split()[-1] if authors and authors[0] else "Unknown"
    arxiv_suffix = paper.arxiv_id.replace(".", "").replace("/", "_")[-8:] if paper.arxiv_id else "unknown"
    year = paper.published_date.year if paper.published_date else "0000"
    cite_key = f"{first_author_last.lower()}{year}_{arxiv_suffix}"

    categories = ""
    if paper.categories:
        categories = ", ".join(paper.categories) if isinstance(paper.categories, list) else str(paper.categories)

    abstract_clean = _escape_bibtex((paper.abstract or "").replace("\n", " ").strip())
    title_clean = _escape_bibtex(paper.title or "Untitled")

    primary_class = "cs.AI"
    if paper.categories:
        if isinstance(paper.categories, list) and paper.categories:
            primary_class = paper.categories[0]
        elif isinstance(paper.categories, str):
            primary_class = paper.categories

    return (
        f"@article{{{cite_key},\n"
        f"  title     = {{{title_clean}}},\n"
        f"  author    = {{{bibtex_authors}}},\n"
        f"  year      = {{{year}}},\n"
        f"  eprint    = {{{paper.arxiv_id or ''}}},\n"
        f"  archivePrefix = {{arXiv}},\n"
        f"  primaryClass  = {{{primary_class}}},\n"
        f"  url       = {{{paper.pdf_url or ''}}},\n"
        f"  abstract  = {{{abstract_clean}}},\n"
        f"  keywords  = {{{categories}}}\n"
        f"}}"
    )


def _format_ris(paper: Any) -> str:
    """Format a paper as a RIS (Research Information Systems) entry."""
    lines = [
        "TY  - JOUR",
        f"TI  - {paper.title or 'Untitled'}",
    ]

    authors = paper.authors if isinstance(paper.authors, list) else [paper.authors] if paper.authors else ["Unknown"]
    for author in authors:
        lines.append(f"AU  - {author}")

    if paper.published_date:
        lines.append(f"PY  - {paper.published_date.year}")
        lines.append(f"DA  - {paper.published_date.strftime('%Y/%m/%d')}")

    lines.append(f"UR  - {paper.pdf_url or ''}")
    lines.append(f"ER  - ")

    abstract = (paper.abstract or "").replace("\n", " ").strip()
    lines.insert(-1, f"AB  - {abstract}")

    if paper.categories:
        cats = ", ".join(paper.categories) if isinstance(paper.categories, list) else str(paper.categories)
        lines.insert(-1, f"KW  - {cats}")

    lines.insert(-1, f"AN  - {paper.arxiv_id or ''}")

    return "\n".join(lines)


@router.get(
    "/papers/export/batch",
    response_class=PlainTextResponse,
    summary="Export multiple paper citations as a single BibTeX file",
)
def export_citations_batch(
    format: str = Query("bibtex", enum=["bibtex", "ris"]),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: SessionDep = None,
    _key: APIKeyDep = None,
) -> PlainTextResponse:
    """Export multiple paper citations in a single file.

    Useful for bulk-importing references into a reference manager.
    """
    repo = PaperRepository(db)
    papers = repo.get_all(limit=limit, offset=offset)

    if not papers:
        raise HTTPException(status_code=404, detail="No papers found")

    if format == "bibtex":
        entries = [_format_bibtex(p) for p in papers]
        content = "\n\n".join(entries)
        media_type = "application/x-bibtex"
        filename = "arxiv_papers.bib"
    else:
        entries = [_format_ris(p) for p in papers]
        content = "\n\n".join(entries)
        media_type = "application/x-research-info-systems"
        filename = "arxiv_papers.ris"

    return PlainTextResponse(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get(
    "/papers/{arxiv_id}/export",
    response_class=PlainTextResponse,
    summary="Export paper citation in BibTeX or RIS format",
)
def export_citation(
    arxiv_id: str,
    format: str = Query("bibtex", enum=["bibtex", "ris"], description="Citation format"),
    db: SessionDep = None,
    _key: APIKeyDep = None,
) -> PlainTextResponse:
    """Export a single paper's citation metadata.

    Returns a plain-text response in the requested format suitable for
    import into reference managers (Zotero, Mendeley, etc.).
    """
    arxiv_id = validate_arxiv_id(arxiv_id)
    repo = PaperRepository(db)
    paper = repo.get_by_arxiv_id(arxiv_id)

    if not paper:
        raise HTTPException(status_code=404, detail=f"Paper {arxiv_id} not found")

    if format == "bibtex":
        content = _format_bibtex(paper)
        media_type = "application/x-bibtex"
        filename = f"{arxiv_id}.bib"
    else:
        content = _format_ris(paper)
        media_type = "application/x-research-info-systems"
        filename = f"{arxiv_id}.ris"

    return PlainTextResponse(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
