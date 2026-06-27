import socket
import pytest
from src.config import get_settings
from src.services.arxiv.factory import make_arxiv_client
from src.services.opensearch.factory import make_opensearch_client


def is_offline(host="export.arxiv.org", port=80, timeout=2) -> bool:
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return False
    except Exception:
        return True


@pytest.mark.skipif(is_offline(), reason="arXiv API is not reachable (offline test environment)")
async def test_arxiv_client_basic():
    client = make_arxiv_client()

    papers = await client.fetch_papers_with_query("cat:cs.AI", max_results=1)

    assert isinstance(papers, list)


def test_opensearch_client_health():
    client = make_opensearch_client()

    health = client.health_check()
    assert isinstance(health, bool)


def test_settings_loading():
    settings = get_settings()

    assert hasattr(settings, "app_version")
    assert hasattr(settings, "service_name")
    assert hasattr(settings, "environment")
