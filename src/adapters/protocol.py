"""Platform adapter protocol definitions.

This module defines the abstract interfaces that platform adapters must implement.
"""

from datetime import datetime
from typing import Optional, Protocol, runtime_checkable

from pydantic import BaseModel


class PlatformPost(BaseModel):
    """Generic post model that works across platforms.

    This is the common data structure that brain.py works with.
    Platform-specific adapters convert their native post format to this.
    """

    id: str
    text: Optional[str] = None
    timestamp: datetime
    username: Optional[str] = None  # Author's username
    permalink: Optional[str] = None
    is_reply: bool = False
    replied_to_id: Optional[str] = None
    platform: str = "unknown"
    media_type: Optional[str] = None  # "TEXT", "IMAGE", "VIDEO", etc.

    # Platform-specific metadata (optional)
    raw_data: Optional[dict] = None


class PlatformUser(BaseModel):
    """Generic user model."""

    id: str
    username: str
    display_name: Optional[str] = None
    bio: Optional[str] = None
    avatar_url: Optional[str] = None


class SearchResult(BaseModel):
    """Search result container."""

    posts: list[PlatformPost]
    has_more: bool = False
    next_cursor: Optional[str] = None


@runtime_checkable
class PlatformAdapter(Protocol):
    """Protocol for platform adapters.

    Implement this protocol to add support for a new social platform.
    See ThreadsAdapter for a reference implementation.
    """

    # Lifecycle methods
    async def open(self) -> None:
        """Initialize the adapter (e.g., open HTTP connections)."""
        ...

    async def close(self) -> None:
        """Clean up resources (e.g., close HTTP connections)."""
        ...

    # Post operations
    async def post(self, content: str) -> str:
        """Create a new post.

        Args:
            content: The text content to post.

        Returns:
            The ID of the created post.
        """
        ...

    async def reply(self, post_id: str, content: str) -> str:
        """Reply to an existing post.

        Args:
            post_id: The ID of the post to reply to.
            content: The text content of the reply.

        Returns:
            The ID of the created reply.
        """
        ...

    async def get_post(self, post_id: str) -> PlatformPost:
        """Get a specific post by ID.

        Args:
            post_id: The ID of the post to retrieve.

        Returns:
            The post data.
        """
        ...

    # Mentions / Replies
    async def get_mentions(
        self,
        max_posts: int = 10,
        max_replies_per_post: int = 10,
    ) -> list[PlatformPost]:
        """Get mentions or replies to the authenticated user.

        Args:
            max_posts: Maximum number of own posts to check.
            max_replies_per_post: Maximum replies to fetch per post.

        Returns:
            List of posts that mention or reply to the user.
        """
        ...

    # Search (optional capability)
    async def search(
        self,
        query: str,
        limit: int = 25,
    ) -> SearchResult:
        """Search for posts matching a query.

        Note: Not all platforms support search. Implementations may raise
        NotImplementedError if search is not available.

        Args:
            query: Search query string.
            limit: Maximum number of results.

        Returns:
            Search results.
        """
        ...

    # Rate limiting
    async def can_post(self) -> bool:
        """Check if posting is allowed (rate limit check).

        Returns:
            True if posting is allowed.
        """
        ...

    async def can_reply(self) -> bool:
        """Check if replying is allowed (rate limit check).

        Returns:
            True if replying is allowed.
        """
        ...

    # User profile
    async def get_user_profile(self, user_id: Optional[str] = None) -> PlatformUser:
        """Get user profile information.

        Args:
            user_id: User ID to fetch. If None, fetches the authenticated user.

        Returns:
            User profile data.
        """
        ...
