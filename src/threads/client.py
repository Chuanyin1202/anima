"""Threads API Client implementation.

Official Threads API documentation:
https://developers.facebook.com/docs/threads

Rate Limits:
- 250 posts per 24 hours
- 1,000 replies per 24 hours
- 500 search queries per 7 days
"""

import asyncio
from datetime import datetime
from typing import Optional

import httpx
import structlog

from .models import (
    MediaType,
    Post,
    PublishRequest,
    RateLimitStatus,
    Reply,
    SearchResult,
    User,
)

logger = structlog.get_logger()


class ThreadsAPIError(Exception):
    """Base exception for Threads API errors."""

    def __init__(self, message: str, status_code: Optional[int] = None, error_code: Optional[str] = None):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(self.message)


class RateLimitExceeded(ThreadsAPIError):
    """Rate limit exceeded exception."""

    pass


class ThreadsClient:
    """Async client for Threads API.

    Usage:
        async with ThreadsClient(access_token, user_id) as client:
            posts = await client.get_user_posts()
            await client.reply_to_post(post_id, "Hello!")
    """

    BASE_URL = "https://graph.threads.net/v1.0"

    def __init__(
        self,
        access_token: str,
        user_id: str,
        timeout: float = 30.0,
    ):
        self.access_token = access_token
        self.user_id = user_id
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def open(self) -> None:
        """Initialize the underlying HTTP client if not already open."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={"Authorization": f"Bearer {self.access_token}"},
            )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "ThreadsClient":
        await self.open()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")
        return self._client

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[dict] = None,
        json: Optional[dict] = None,
    ) -> dict:
        """Make an API request with error handling."""
        url = f"{self.BASE_URL}/{endpoint}"
        params = params or {}
        params["access_token"] = self.access_token

        logger.debug("threads_api_request", method=method, endpoint=endpoint)

        try:
            response = await self.client.request(
                method=method,
                url=url,
                params=params,
                json=json,
            )

            if response.status_code == 429:
                raise RateLimitExceeded(
                    "Rate limit exceeded",
                    status_code=429,
                )

            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            error_data = {}
            try:
                error_data = e.response.json() if e.response.content else {}
            except ValueError:
                error_data = {"raw": e.response.text}

            error_msg = error_data.get("error", {}).get("message", str(e))
            error_code = error_data.get("error", {}).get("code")
            raise ThreadsAPIError(
                message=error_msg,
                status_code=e.response.status_code,
                error_code=error_code,
            ) from e

    # =========================================================================
    # User Profile
    # =========================================================================

    async def get_user_profile(self, user_id: Optional[str] = None) -> User:
        """Get user profile information."""
        user_id = user_id or self.user_id
        data = await self._request(
            "GET",
            user_id,
            params={
                "fields": "id,username,name,threads_profile_picture_url,threads_biography"
            },
        )
        return User(**data)

    # =========================================================================
    # Posts
    # =========================================================================

    async def get_user_posts(
        self,
        limit: int = 25,
        since: Optional[datetime] = None,
    ) -> list[Post]:
        """Get user's own posts."""
        params = {
            "fields": "id,media_type,text,timestamp,permalink,username,is_quote_post,shortcode",
            "limit": limit,
        }
        if since:
            params["since"] = since.isoformat()

        data = await self._request("GET", f"{self.user_id}/threads", params=params)

        posts = []
        for item in data.get("data", []):
            item["timestamp"] = datetime.fromisoformat(
                item["timestamp"].replace("Z", "+00:00")
            )
            posts.append(Post(**item))

        return posts

    async def get_post(self, post_id: str) -> Post:
        """Get a specific post by ID."""
        data = await self._request(
            "GET",
            post_id,
            params={
                "fields": "id,media_type,text,timestamp,permalink,username,is_quote_post"
            },
        )
        data["timestamp"] = datetime.fromisoformat(
            data["timestamp"].replace("Z", "+00:00")
        )
        return Post(**data)

    async def get_post_replies(self, post_id: str, limit: int = 25) -> list[Reply]:
        """Get replies to a post."""
        data = await self._request(
            "GET",
            f"{post_id}/replies",
            params={
                "fields": "id,text,timestamp,username",
                "limit": limit,
            },
        )

        replies = []
        for item in data.get("data", []):
            item["timestamp"] = datetime.fromisoformat(
                item["timestamp"].replace("Z", "+00:00")
            )
            item["replied_to_id"] = post_id
            replies.append(Reply(**item))

        return replies

    # =========================================================================
    # Publishing
    # =========================================================================

    async def create_post(self, text: str) -> str:
        """Create a new text post.

        Returns the published post ID.
        """
        # Step 1: Create media container
        container_data = await self._request(
            "POST",
            f"{self.user_id}/threads",
            params={
                "media_type": "TEXT",
                "text": text,
            },
        )
        container_id = container_data["id"]

        # Step 2: Publish the container
        publish_data = await self._request(
            "POST",
            f"{self.user_id}/threads_publish",
            params={"creation_id": container_id},
        )

        logger.info("post_created", post_id=publish_data["id"])
        return publish_data["id"]

    async def reply_to_post(self, post_id: str, text: str) -> str:
        """Reply to an existing post.

        Returns the reply post ID.
        """
        # Step 1: Create reply container
        container_data = await self._request(
            "POST",
            f"{self.user_id}/threads",
            params={
                "media_type": "TEXT",
                "text": text,
                "reply_to_id": post_id,
            },
        )
        container_id = container_data["id"]

        # Step 2: Publish the reply
        publish_data = await self._request(
            "POST",
            f"{self.user_id}/threads_publish",
            params={"creation_id": container_id},
        )

        logger.info("reply_created", reply_id=publish_data["id"], parent_id=post_id)
        return publish_data["id"]

    # =========================================================================
    # Search & Discovery
    # =========================================================================

    async def search_posts(
        self,
        query: str,
        limit: int = 25,
    ) -> SearchResult:
        """Search for public posts by keyword/topic.

        Note: This endpoint may require additional permissions and is subject to
        500 queries per 7 days rate limit.
        """
        data = await self._request(
            "GET",
            "search",
            params={
                "q": query,
                "type": "threads",
                "limit": limit,
            },
        )

        posts = []
        for item in data.get("data", []):
            item["timestamp"] = datetime.fromisoformat(
                item["timestamp"].replace("Z", "+00:00")
            )
            posts.append(Post(**item))

        return SearchResult(
            posts=posts,
            has_more=bool(data.get("paging", {}).get("next")),
            next_cursor=data.get("paging", {}).get("cursors", {}).get("after"),
        )

    # =========================================================================
    # Rate Limiting
    # =========================================================================

    async def get_rate_limit_status(self) -> RateLimitStatus:
        """Get current rate limit status."""
        data = await self._request(
            "GET",
            f"{self.user_id}/threads_publishing_limit",
            params={"fields": "quota_usage,config"},
        )

        quota_data = data.get("data", [{}])[0]
        config = quota_data.get("config", {})

        return RateLimitStatus(
            quota_usage=quota_data.get("quota_usage", 0),
            quota_total=config.get("quota_total", 250),
            reply_quota_usage=quota_data.get("reply_quota_usage", 0),
            reply_quota_total=config.get("reply_quota_total", 1000),
        )

    async def can_publish(self) -> bool:
        """Check if we can still publish (under rate limit)."""
        status = await self.get_rate_limit_status()
        return status.quota_usage < status.quota_total

    async def can_reply(self) -> bool:
        """Check if we can still reply (under rate limit)."""
        status = await self.get_rate_limit_status()
        return status.reply_quota_usage < status.reply_quota_total
