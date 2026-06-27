"""Tests for the OllamaClient service."""

import httpx
import sys
import types
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


from src.config import Settings
from src.exceptions import OllamaConnectionError, OllamaException
from src.services.ollama.client import OllamaClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_settings(**overrides) -> Settings:
    defaults = {
        "ollama_host": "http://localhost:11434",
        "ollama_timeout": 300,
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _mock_response(status_code: int = 200, json_data: dict | None = None):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    return resp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def settings():
    return _make_settings()


@pytest.fixture
def client(settings: Settings):
    return OllamaClient(settings)


# ---------------------------------------------------------------------------
# Tests – generate
# ---------------------------------------------------------------------------

class TestGenerateSuccess:
    async def test_generate_success(self, client: OllamaClient):
        expected = {
            "model": "llama3.2",
            "response": "Hello, world!",
            "prompt_eval_count": 10,
            "eval_count": 5,
            "total_duration": 1_000_000_000,
        }
        mock_resp = _mock_response(200, expected)

        mock_httpx = AsyncMock()
        mock_httpx.post.return_value = mock_resp
        mock_httpx.is_closed = False

        with patch.object(client, "_get_client", return_value=mock_httpx):
            result = await client.generate(model="llama3.2", prompt="Say hi")

        assert result is not None
        assert result["response"] == "Hello, world!"
        assert result["usage_metadata"]["prompt_tokens"] == 10
        assert result["usage_metadata"]["completion_tokens"] == 5
        assert result["usage_metadata"]["total_tokens"] == 15
        assert result["usage_metadata"]["latency_ms"] == 1000.0


# ---------------------------------------------------------------------------
# Tests – generate retries on failure
# ---------------------------------------------------------------------------

class TestGenerateRetriesOnFailure:
    async def test_generate_retries_on_failure(self, client: OllamaClient):
        fail_resp = _mock_response(500, {"error": "internal error"})

        mock_httpx = AsyncMock()
        mock_httpx.post.return_value = fail_resp
        mock_httpx.is_closed = False

        with patch.object(client, "_get_client", return_value=mock_httpx):
            with pytest.raises(OllamaException):
                await client.generate(model="llama3.2", prompt="fail")

        # Circuit breaker is configured for 3 retries (4 total attempts)
        assert mock_httpx.post.call_count == 4

    async def test_generate_connect_error_raises(self, client: OllamaClient):
        mock_httpx = AsyncMock()
        mock_httpx.post.side_effect = httpx.ConnectError("refused")
        mock_httpx.is_closed = False

        with patch.object(client, "_get_client", return_value=mock_httpx):
            with pytest.raises(OllamaConnectionError):
                await client.generate(model="llama3.2", prompt="fail")


# ---------------------------------------------------------------------------
# Tests – health_check
# ---------------------------------------------------------------------------

class TestHealthCheckReturnsStatus:
    async def test_health_check_returns_status(self, client: OllamaClient):
        mock_resp = _mock_response(200, {"version": "0.3.0"})

        mock_httpx = AsyncMock()
        mock_httpx.get.return_value = mock_resp
        mock_httpx.is_closed = False

        with patch.object(client, "_get_client", return_value=mock_httpx):
            status = await client.health_check()

        assert status["status"] == "healthy"
        assert status["version"] == "0.3.0"
        assert "message" in status

    async def test_health_check_connection_error(self, client: OllamaClient):
        mock_httpx = AsyncMock()
        mock_httpx.get.side_effect = httpx.ConnectError("refused")
        mock_httpx.is_closed = False

        with patch.object(client, "_get_client", return_value=mock_httpx):
            with pytest.raises(OllamaConnectionError):
                await client.health_check()

    async def test_health_check_non_200_raises(self, client: OllamaClient):
        mock_resp = _mock_response(503, {})
        mock_httpx = AsyncMock()
        mock_httpx.get.return_value = mock_resp
        mock_httpx.is_closed = False

        with patch.object(client, "_get_client", return_value=mock_httpx):
            with pytest.raises(OllamaException):
                await client.health_check()
