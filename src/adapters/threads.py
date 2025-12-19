"""Threads platform adapter.

This module implements the PlatformAdapter protocol for Meta's Threads platform.
"""

from typing import Optional

import structlog

from ..threads import ThreadsClient
from ..threads.models import Post as ThreadsPost
from ..threads.models import SearchResult as ThreadsSearchResult
from .protocol import PlatformAdapter, PlatformPost, PlatformUser, SearchResult

logger = structlog.get_logger()


class ThreadsAdapter(PlatformAdapter):
    """Adapter for Threads platform.

    Wraps ThreadsClient to implement the generic PlatformAdapter interface.

    Usage:
        client = ThreadsClient(access_token, user_id)
        adapter = ThreadsAdapter(client)

        async with adapter:
            posts = await adapter.get_mentions()
            await adapter.reply(post_id, "Hello!")
    """

    def __init__(self, client: ThreadsClient):
        """Initialize the adapter.

        Args:
            client: Configured ThreadsClient instance.
        """
        self._client = client

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def open(self) -> None:
        """Initialize the underlying HTTP client."""
        await self._client.open()

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()

    async def __aenter__(self) -> "ThreadsAdapter":
        await self.open()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    # =========================================================================
    # Post Operations
    # =========================================================================

    async def post(self, content: str) -> str:
        """Create a new post on Threads.

        Args:
            content: The text content to post.

        Returns:
            The ID of the created post.
        """
        return await self._client.create_post(content)

    async def reply(self, post_id: str, content: str) -> str:
        """Reply to an existing Threads post.

        Args:
            post_id: The ID of the post to reply to.
            content: The text content of the reply.

        Returns:
            The ID of the created reply.
        """
        return await self._client.reply_to_post(post_id, content)

    async def get_post(self, post_id: str) -> PlatformPost:
        """Get a specific post by ID.

        Args:
            post_id: The ID of the post to retrieve.

        Returns:
            The post data as a PlatformPost.
        """
        threads_post = await self._client.get_post(post_id)
        return self._convert_post(threads_post)

    # =========================================================================
    # Mentions / Replies
    # =========================================================================

    async def get_mentions(
        self,
        max_posts: int = 10,
        max_replies_per_post: int = 10,
    ) -> list[PlatformPost]:
        """Get replies to the authenticated user's posts.

        On Threads, this retrieves replies to the user's recent posts.

        Args:
            max_posts: Maximum number of own posts to check.
            max_replies_per_post: Maximum replies to fetch per post.

        Returns:
            List of posts that reply to the user.
        """
        threads_posts = await self._client.get_replies_to_my_posts(
            max_posts=max_posts,
            max_replies_per_post=max_replies_per_post,
        )
        return [self._convert_post(p) for p in threads_posts]

    # =========================================================================
    # Search
    # =========================================================================

    async def search(
        self,
        query: str,
        limit: int = 25,
    ) -> SearchResult:
        """Search for posts matching a query.

        Note: Requires threads_keyword_search permission from Meta.

        Args:
            query: Search query string.
            limit: Maximum number of results.

        Returns:
            Search results.
        """
        threads_result = await self._client.search_posts(query=query, limit=limit)
        return SearchResult(
            posts=[self._convert_post(p) for p in threads_result.posts],
            has_more=threads_result.has_more,
            next_cursor=threads_result.next_cursor,
        )

    # =========================================================================
    # Rate Limiting
    # =========================================================================

    async def can_post(self) -> bool:
        """Check if posting is allowed (within rate limit).

        Returns:
            True if posting is allowed.
        """
        return await self._client.can_publish()

    async def can_reply(self) -> bool:
        """Check if replying is allowed (within rate limit).

        Returns:
            True if replying is allowed.
        """
        return await self._client.can_reply()

    # =========================================================================
    # User Profile
    # =========================================================================

    async def get_user_profile(self, user_id: Optional[str] = None) -> PlatformUser:
        """Get user profile information.

        Args:
            user_id: User ID to fetch. If None, fetches the authenticated user.

        Returns:
            User profile data.
        """
        threads_user = await self._client.get_user_profile(user_id)
        return PlatformUser(
            id=threads_user.id,
            username=threads_user.username,
            display_name=threads_user.name,
            bio=threads_user.threads_biography,
            avatar_url=threads_user.threads_profile_picture_url,
        )

    # =========================================================================
    # Helpers
    # =========================================================================

    def _convert_post(self, threads_post: ThreadsPost) -> PlatformPost:
        """Convert a Threads Post to generic PlatformPost.

        Args:
            threads_post: The Threads-specific post object.

        Returns:
            A generic PlatformPost.
        """
        return PlatformPost(
            id=threads_post.id,
            text=threads_post.text,
            timestamp=threads_post.timestamp,
            username=threads_post.username,
            permalink=threads_post.permalink,
            is_reply=threads_post.is_reply or False,
            replied_to_id=threads_post.replied_to_id,
            platform="threads",
            media_type=threads_post.media_type.value if threads_post.media_type else None,
            raw_data=threads_post.model_dump(),
        )
