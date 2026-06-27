"""Tests for paper sync service."""

from unittest.mock import MagicMock, patch
import pytest
from src.services.paper_sync import PaperSyncService
from src.schemas.arxiv.paper import PaperCreate


@pytest.fixture
def mock_opensearch():
    """Mock OpenSearch client."""
    mock = MagicMock()
    mock.client = MagicMock()
    mock.index_name = "arxiv-papers"
    return mock


@pytest.fixture
def mock_session():
    """Mock database session."""
    return MagicMock()


@pytest.fixture
def paper_sync_service(mock_opensearch, mock_session):
    """Create PaperSyncService instance."""
    # Patch PaperRepository to avoid real DB dependency inside tests
    with patch("src.services.paper_sync.PaperRepository") as mock_repo_cls:
        service = PaperSyncService(mock_opensearch, mock_session)
        service.mock_repo = mock_repo_cls.return_value
        yield service


class TestPaperSyncService:
    """Tests for PaperSyncService."""

    def test_sync_all_success(self, paper_sync_service, mock_opensearch, mock_session):
        """Test syncing all papers from OpenSearch to database successfully."""
        # Setup OpenSearch search hits
        mock_opensearch.client.search.side_effect = [
            {
                "hits": {
                    "hits": [
                        {
                            "_source": {
                                "arxiv_id": "2301.00001",
                                "title": "Paper 1",
                                "authors": ["Author A"],
                                "abstract": "Abstract 1",
                                "categories": ["cs.AI"],
                                "published_date": "2023-01-01T00:00:00",
                                "pdf_url": "https://arxiv.org/pdf/2301.00001.pdf",
                            }
                        },
                        {
                            "_source": {
                                "arxiv_id": "2301.00002",
                                "title": "Paper 2",
                                "authors": ["Author B"],
                                "abstract": "Abstract 2",
                                "categories": ["cs.LG"],
                                "published_date": "2023-01-02T00:00:00",
                                "pdf_url": "https://arxiv.org/pdf/2301.00002.pdf",
                            }
                        },
                    ]
                }
            },
            # Return empty hits on second call to stop the pagination loop
            {"hits": {"hits": []}}
        ]

        result = paper_sync_service.sync_all(batch_size=2)

        # Assert correct count of synced items
        assert result == {"synced": 2, "failed": 0, "total": 2}
        # Check that PaperRepository.upsert was called for each paper
        assert paper_sync_service.mock_repo.upsert.call_count == 2
        # Check commit was called
        mock_session.commit.assert_called_once()

    def test_sync_all_empty(self, paper_sync_service, mock_opensearch):
        """Test syncing when OpenSearch is empty."""
        mock_opensearch.client.search.return_value = {"hits": {"hits": []}}

        result = paper_sync_service.sync_all()

        assert result == {"synced": 0, "failed": 0, "total": 0}
        assert paper_sync_service.mock_repo.upsert.call_count == 0

    def test_sync_single_success(self, paper_sync_service, mock_opensearch, mock_session):
        """Test syncing a single paper from OpenSearch to database successfully."""
        mock_opensearch.client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "arxiv_id": "2301.00001",
                            "title": "Paper 1",
                            "authors": ["Author A"],
                            "abstract": "Abstract 1",
                            "categories": ["cs.AI"],
                            "published_date": "2023-01-01T00:00:00",
                            "pdf_url": "https://arxiv.org/pdf/2301.00001.pdf",
                        }
                    }
                ]
            }
        }

        result = paper_sync_service.sync_single("2301.00001")

        assert result == "2301.00001"
        paper_sync_service.mock_repo.upsert.assert_called_once()
        mock_session.commit.assert_called_once()

    def test_sync_single_not_found(self, paper_sync_service, mock_opensearch):
        """Test syncing a single paper when not found in OpenSearch."""
        mock_opensearch.client.search.return_value = {"hits": {"hits": []}}

        result = paper_sync_service.sync_single("nonexistent")

        assert result is None
        assert paper_sync_service.mock_repo.upsert.call_count == 0
