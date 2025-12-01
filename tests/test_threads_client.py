"""Tests for Threads API Client."""

from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from src.threads.client import ThreadsClient, ThreadsAPIError, RateLimitExceeded
from src.threads.models import MediaType, Post, RateLimitStatus


class TestThreadsModels:
    """Tests for Threads data models."""

    def test_post_model(self):
        """Test Post model creation."""
        post = Post(
            id="123456",
            media_type=MediaType.TEXT,
            text="Hello, world!",
            timestamp=datetime.now(),
            username="test_user",
        )

        assert post.id == "123456"
        assert post.media_type == MediaType.TEXT
        assert post.text == "Hello, world!"

    def test_rate_limit_status(self):
        """Test RateLimitStatus model."""
        status = RateLimitStatus(
            quota_usage=10,
            quota_total=250,
            reply_quota_usage=50,
            reply_quota_total=1000,
        )

        assert status.quota_usage == 10
        assert status.quota_total == 250


class TestThreadsClient:
    """Tests for ThreadsClient."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        return ThreadsClient(
            access_token="test_token",
            user_id="test_user_id",
        )

    def test_client_initialization(self, client):
        """Test client initialization."""
        assert client.access_token == "test_token"
        assert client.user_id == "test_user_id"
        assert client._client is None  # Not initialized until context manager

    @pytest.mark.asyncio
    async def test_client_context_manager(self, client):
        """Test client context manager."""
        async with client:
            assert client._client is not None

        # Client should be closed after context
        # Note: We can't easily verify this without mocking

    @pytest.mark.asyncio
    async def test_client_not_initialized_error(self, client):
        """Test error when accessing client outside context."""
        with pytest.raises(RuntimeError, match="not initialized"):
            _ = client.client

    @pytest.mark.asyncio
    async def test_can_publish(self, client):
        """Test can_publish method."""
        with patch.object(client, "get_rate_limit_status") as mock_status:
            mock_status.return_value = RateLimitStatus(
                quota_usage=10,
                quota_total=250,
            )

            async with client:
                result = await client.can_publish()

            assert result is True

    @pytest.mark.asyncio
    async def test_can_publish_at_limit(self, client):
        """Test can_publish when at limit."""
        with patch.object(client, "get_rate_limit_status") as mock_status:
            mock_status.return_value = RateLimitStatus(
                quota_usage=250,
                quota_total=250,
            )

            async with client:
                result = await client.can_publish()

            assert result is False


class TestThreadsAPIErrors:
    """Tests for API error handling."""

    def test_threads_api_error(self):
        """Test ThreadsAPIError creation."""
        error = ThreadsAPIError(
            message="Something went wrong",
            status_code=400,
            error_code="INVALID_REQUEST",
        )

        assert str(error) == "Something went wrong"
        assert error.status_code == 400
        assert error.error_code == "INVALID_REQUEST"

    def test_rate_limit_exceeded(self):
        """Test RateLimitExceeded error."""
        error = RateLimitExceeded(
            message="Rate limit exceeded",
            status_code=429,
        )

        assert isinstance(error, ThreadsAPIError)
        assert error.status_code == 429
