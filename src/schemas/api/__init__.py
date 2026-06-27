import re

from fastapi import HTTPException

# Validates arXiv IDs: digits, optional .vN suffix, optional leading category prefix
_ARXIV_ID_PATTERN = re.compile(r"^[a-zA-Z0-9]+(\.[0-9]+)?(v[0-9]+)?$")


def validate_arxiv_id(arxiv_id: str) -> str:
    """Validate and return a safe arXiv ID, raising 422 if invalid."""
    if not arxiv_id or not arxiv_id.strip():
        raise HTTPException(status_code=422, detail="arxiv_id cannot be empty")
    arxiv_id = arxiv_id.strip()
    if len(arxiv_id) > 64:
        raise HTTPException(status_code=422, detail="arxiv_id too long (max 64 chars)")
    if not _ARXIV_ID_PATTERN.match(arxiv_id):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid arxiv_id format: {arxiv_id!r}. Expected pattern like 2301.12345 or cs.AI/2301.12345",
        )
    return arxiv_id
