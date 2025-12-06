"""External post provider protocol.

This allows plugging in additional post sources (e.g., Apify, RSS, JSON)
without coupling the core agent to external dependencies.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import Post


@runtime_checkable
class ExternalPostProvider(Protocol):
    """Protocol for external post sources."""

    name: str  # Provider name for logging

    async def fetch_posts(self, max_items: int = 50) -> list[Post]:
        """Fetch posts from external source.

        Should return empty list on failure; must not raise.
        """
        ...
