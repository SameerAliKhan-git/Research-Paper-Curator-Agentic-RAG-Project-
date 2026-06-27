"""Tests for API key authentication service."""

import json
from unittest.mock import AsyncMock, MagicMock
import pytest
from src.services.auth.api_key_service import APIKeyService, APIKeyMetadata


@pytest.fixture
def mock_redis():
    """Mock async Redis client."""
    mock = AsyncMock()
    # register_script is synchronous in redis-py, returning a script object
    from unittest.mock import MagicMock
    mock.register_script = MagicMock(return_value=AsyncMock())
    return mock


@pytest.fixture
def api_key_service(mock_redis):
    """Create APIKeyService instance."""
    return APIKeyService(mock_redis)


class TestAPIKeyService:
    """Tests for APIKeyService."""

    @pytest.mark.asyncio
    async def test_create_key(self, api_key_service, mock_redis):
        """Test registering a new API key."""
        raw_key = "test-raw-key-12345"
        user_id = "test_user"
        
        result = await api_key_service.create_key(
            raw_key=raw_key,
            user_id=user_id,
            tier="admin",
            rate_limit=100,
            daily_quota=1000,
            tenants=["default"],
        )

        assert result["user_id"] == user_id
        assert result["tier"] == "admin"
        assert result["rate_limit"] == 100
        assert result["daily_quota"] == 1000
        assert result["tenants"] == ["default"]
        assert "key_hash" in result

        # Verify Redis set was called with correct key and payload
        mock_redis.set.assert_called_once()
        call_key = mock_redis.set.call_args[0][0]
        assert call_key.startswith("auth:keys:")

    @pytest.mark.asyncio
    async def test_validate_key_valid(self, api_key_service, mock_redis):
        """Test validating an active valid key."""
        meta_dict = {
            "user_id": "user123",
            "tier": "standard",
            "rate_limit": 60,
            "daily_quota": 1000,
            "quota_remaining": 1000,
            "enabled": True,
            "tenants": ["default"],
        }
        mock_redis.get.return_value = json.dumps(meta_dict)

        metadata = await api_key_service.validate_key("raw-key-value")

        assert metadata is not None
        assert metadata.user_id == "user123"
        assert metadata.tier == "standard"
        assert metadata.rate_limit == 60
        assert metadata.tenants == ["default"]

    @pytest.mark.asyncio
    async def test_validate_key_invalid(self, api_key_service, mock_redis):
        """Test validating a non-existent key."""
        mock_redis.get.return_value = None

        metadata = await api_key_service.validate_key("invalid-key")
        assert metadata is None

    @pytest.mark.asyncio
    async def test_validate_key_disabled(self, api_key_service, mock_redis):
        """Test validating a disabled key."""
        meta_dict = {
            "user_id": "user123",
            "enabled": False,
        }
        mock_redis.get.return_value = json.dumps(meta_dict)

        metadata = await api_key_service.validate_key("disabled-key")
        assert metadata is None

    @pytest.mark.asyncio
    async def test_check_rate_limit_allowed(self, api_key_service, mock_redis):
        """Test check_rate_limit when allowed."""
        # The Lua script returns 1 if allowed
        api_key_service._rate_limit_script.return_value = 1

        allowed = await api_key_service.check_rate_limit("some_key_hash", 60)
        assert allowed is True

    @pytest.mark.asyncio
    async def test_check_rate_limit_exceeded(self, api_key_service, mock_redis):
        """Test check_rate_limit when limit exceeded."""
        # The Lua script returns 0 if rate limited
        api_key_service._rate_limit_script.return_value = 0

        allowed = await api_key_service.check_rate_limit("some_key_hash", 60)
        assert allowed is False

    @pytest.mark.asyncio
    async def test_decrement_quota_success(self, api_key_service, mock_redis):
        """Test decrementing quota successfully."""
        meta_dict = {
            "user_id": "user123",
            "quota_remaining": 100,
        }
        mock_redis.get.return_value = json.dumps(meta_dict)
        # The decr quota Lua script returns remaining quota
        api_key_service._decr_quota_script.return_value = 99

        remaining = await api_key_service.decrement_quota("some_key_hash")
        assert remaining == 99

    @pytest.mark.asyncio
    async def test_decrement_quota_unlimited(self, api_key_service, mock_redis):
        """Test decrementing quota when key has unlimited quota (-1)."""
        meta_dict = {
            "user_id": "user123",
            "quota_remaining": -1,
        }
        mock_redis.get.return_value = json.dumps(meta_dict)

        remaining = await api_key_service.decrement_quota("some_key_hash")
        assert remaining == -1
        # Lua script shouldn't be executed for unlimited keys
        api_key_service._decr_quota_script.assert_not_called()

    @pytest.mark.asyncio
    async def test_revoke_key_success(self, api_key_service, mock_redis):
        """Test revoking an API key."""
        meta_dict = {
            "user_id": "user123",
            "enabled": True,
        }
        mock_redis.get.return_value = json.dumps(meta_dict)

        result = await api_key_service.revoke_key("my-key")
        assert result is True
        # Verify it updated the key to disabled in Redis
        mock_redis.set.assert_called_once()
        set_payload = json.loads(mock_redis.set.call_args[0][1])
        assert set_payload["enabled"] is False


class TestAPIKeyMetadata:
    """Tests for APIKeyMetadata object."""

    def test_metadata_properties(self):
        """Test APIKeyMetadata attributes."""
        metadata = APIKeyMetadata(
            key_hash="hash123",
            user_id="user123",
            tier="premium",
            rate_limit=120,
            quota_remaining=500,
            tenants=["default", "custom"],
        )

        assert metadata.key_hash == "hash123"
        assert metadata.user_id == "user123"
        assert metadata.tier == "premium"
        assert metadata.rate_limit == 120
        assert metadata.quota_remaining == 500
        assert metadata.tenants == ["default", "custom"]
