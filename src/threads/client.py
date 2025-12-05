"""Threads API Client implementation.

Official Threads API documentation:
https://developers.facebook.com/docs/threads

Rate Limits:
- 250 posts per 24 hours
- 1,000 replies per 24 hours
- 2,200 search queries per 24 hours (requires threads_keyword_search permission)
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

            logger.warning(
                "api_error_details",
                status_code=e.response.status_code,
                error_code=error_code,
                error_msg=error_msg,
                error_data=error_data,
                endpoint=endpoint,
                method=method,
            )

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
        cursor: Optional[str] = None,
    ) -> tuple[list[Post], Optional[str]]:
        """Get user's own posts with pagination support.

        Returns:
            Tuple of (posts, next_cursor). next_cursor is None if no more pages.
        """
        params = {
            "fields": "id,media_type,text,timestamp,permalink,username,is_quote_post,shortcode",
            "limit": limit,
        }
        if since:
            params["since"] = since.isoformat()
        if cursor:
            params["after"] = cursor

        data = await self._request("GET", f"{self.user_id}/threads", params=params)

        posts = []
        for item in data.get("data", []):
            item["timestamp"] = datetime.fromisoformat(
                item["timestamp"].replace("Z", "+00:00")
            )
            posts.append(Post(**item))

        # Get next cursor for pagination
        next_cursor = data.get("paging", {}).get("cursors", {}).get("after")

        return posts, next_cursor

    async def get_all_user_posts(
        self,
        since: Optional[datetime] = None,
        max_posts: int = 500,
    ) -> list[Post]:
        """Get all user's posts by paginating through results.

        Args:
            since: Only get posts after this datetime
            max_posts: Maximum number of posts to retrieve (safety limit)

        Returns:
            List of all posts
        """
        all_posts = []
        cursor = None

        while len(all_posts) < max_posts:
            posts, next_cursor = await self.get_user_posts(
                limit=25,
                since=since,
                cursor=cursor,
            )

            if not posts:
                break

            all_posts.extend(posts)
            logger.info("fetched_posts_page", count=len(posts), total=len(all_posts))

            if not next_cursor:
                break

            cursor = next_cursor
            # Small delay to avoid rate limiting
            await asyncio.sleep(0.5)

        return all_posts

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

    async def get_replies_to_my_posts(
        self,
        max_posts: int = 10,
        max_replies_per_post: int = 10,
    ) -> list[Post]:
        """Get replies to my recent posts (no special permission needed).

        This is an alternative to search_posts() when you don't have
        the threads_keyword_search permission.

        Args:
            max_posts: Maximum number of my posts to check for replies.
            max_replies_per_post: Maximum replies to fetch per post.

        Returns:
            List of Post objects representing replies to my posts.
        """
        all_replies: list[Post] = []

        # Get my recent posts
        my_posts, _ = await self.get_user_posts(limit=max_posts)

        for post in my_posts:
            try:
                replies = await self.get_post_replies(post.id, limit=max_replies_per_post)

                # Convert Reply to Post format for compatibility
                for reply in replies:
                    reply_as_post = Post(
                        id=reply.id,
                        media_type=MediaType.TEXT,
                        text=reply.text,
                        timestamp=reply.timestamp,
                        username=reply.username,
                        is_reply=True,
                        replied_to_id=reply.replied_to_id,
                    )
                    all_replies.append(reply_as_post)

                # Small delay to avoid rate limiting
                await asyncio.sleep(0.2)

            except Exception as e:
                logger.warning("fetch_replies_failed", post_id=post.id, error=str(e))
                continue

        logger.info("replies_fetched", total=len(all_replies), posts_checked=len(my_posts))
        return all_replies

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
        search_type: str = "TOP",
        search_mode: str = "KEYWORD",
        media_type: Optional[str] = None,
        since: Optional[int] = None,
        until: Optional[int] = None,
    ) -> SearchResult:
        """Search for public posts by keyword or hashtag.

        ⚠️ REQUIRES SPECIAL PERMISSION: threads_keyword_search
        This permission requires Meta app review approval.
        If you don't have this permission, use get_replies_to_my_posts() instead.

        Official endpoint: GET /keyword_search
        Docs: https://developers.facebook.com/docs/threads/keyword-search

        Args:
            query: The keyword or hashtag to search for.
            limit: Max results to return (1-100, default 25).
            search_type: "TOP" (default) for popular results, "RECENT" for newest.
            search_mode: "KEYWORD" (default) for keyword search, "TAG" for hashtag.
            media_type: Filter by media type - "TEXT", "IMAGE", or "VIDEO".
            since: Unix timestamp for start date (must be >= 1688540400).
            until: Unix timestamp for end date (must be <= now).

        Returns:
            SearchResult with matching posts.

        Requires:
            - threads_basic permission
            - threads_keyword_search permission (NEEDS APP REVIEW)

        Rate limit: 2,200 queries per 24 hours.
        Note: Queries returning no results don't count against the limit.
        """
        params = {
            "q": query,
            "search_type": search_type,
            "search_mode": search_mode,
            "limit": min(limit, 100),
            "fields": "id,text,media_type,permalink,timestamp,username,has_replies,is_quote_post,is_reply",
        }

        if media_type:
            params["media_type"] = media_type
        if since is not None:
            params["since"] = since
        if until is not None:
            params["until"] = until

        data = await self._request("GET", "keyword_search", params=params)

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
