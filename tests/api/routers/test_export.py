"""Tests for citation export endpoints."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture
def mock_paper():
    """Create a mock paper object."""
    paper = MagicMock()
    paper.arxiv_id = "2301.00001"
    paper.title = "Test Paper Title"
    paper.authors = ["Author One", "Author Two"]
    paper.abstract = "This is a test abstract."
    paper.categories = ["cs.AI", "cs.CL"]
    paper.published_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
    paper.pdf_url = "https://arxiv.org/pdf/2301.00001.pdf"
    return paper


@pytest.fixture
def mock_paper_repository():
    """Mock PaperRepository."""
    with patch("src.routers.export.PaperRepository") as mock_repo:
        yield mock_repo


class TestExportCitation:
    """Tests for single paper citation export."""

    @pytest.mark.anyio
    async def test_export_bibtex(self, client: AsyncClient, mock_paper_repository, mock_paper):
        """Test BibTeX export for a single paper."""
        mock_repo_instance = MagicMock()
        mock_repo_instance.get_by_arxiv_id.return_value = mock_paper
        mock_paper_repository.return_value = mock_repo_instance

        response = await client.get("/api/v1/papers/2301.00001/export?format=bibtex")

        assert response.status_code == 200
        assert "bibtex" in response.headers.get("content-type", "").lower() or "@article" in response.text
        assert "@article{" in response.text
        assert "Test Paper Title" in response.text
        assert "2301.00001" in response.text

    @pytest.mark.anyio
    async def test_export_ris(self, client: AsyncClient, mock_paper_repository, mock_paper):
        """Test RIS export for a single paper."""
        mock_repo_instance = MagicMock()
        mock_repo_instance.get_by_arxiv_id.return_value = mock_paper
        mock_paper_repository.return_value = mock_repo_instance

        response = await client.get("/api/v1/papers/2301.00001/export?format=ris")

        assert response.status_code == 200
        assert "TY  - JOUR" in response.text
        assert "TI  - Test Paper Title" in response.text
        assert "AU  - Author One" in response.text

    @pytest.mark.anyio
    async def test_export_paper_not_found(self, client: AsyncClient, mock_paper_repository):
        """Test export for non-existent paper."""
        mock_repo_instance = MagicMock()
        mock_repo_instance.get_by_arxiv_id.return_value = None
        mock_paper_repository.return_value = mock_repo_instance

        response = await client.get("/api/v1/papers/nonexistent/export")

        assert response.status_code == 404


class TestExportBatch:
    """Tests for batch citation export."""

    @pytest.mark.anyio
    async def test_export_batch_bibtex(self, client: AsyncClient, mock_paper_repository, mock_paper):
        """Test batch BibTeX export."""
        mock_repo_instance = MagicMock()
        mock_repo_instance.get_all.return_value = [mock_paper]
        mock_paper_repository.return_value = mock_repo_instance

        response = await client.get("/api/v1/papers/export/batch?format=bibtex&limit=10")

        assert response.status_code == 200
        assert "@article{" in response.text

    @pytest.mark.anyio
    async def test_export_batch_ris(self, client: AsyncClient, mock_paper_repository, mock_paper):
        """Test batch RIS export."""
        mock_repo_instance = MagicMock()
        mock_repo_instance.get_all.return_value = [mock_paper]
        mock_paper_repository.return_value = mock_repo_instance

        response = await client.get("/api/v1/papers/export/batch?format=ris&limit=10")

        assert response.status_code == 200
        assert "TY  - JOUR" in response.text

    @pytest.mark.anyio
    async def test_export_batch_no_papers(self, client: AsyncClient, mock_paper_repository):
        """Test batch export when no papers exist."""
        mock_repo_instance = MagicMock()
        mock_repo_instance.get_all.return_value = []
        mock_paper_repository.return_value = mock_repo_instance

        response = await client.get("/api/v1/papers/export/batch")

        assert response.status_code == 404
