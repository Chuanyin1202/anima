"""Threads Toolkit provider for fetching external posts.

Designed to consume JSON output from Threads Toolkit (or similar APIs)
and convert them into internal Post objects via ingest_posts.
"""

from __future__ import annotations

import structlog
from typing import Any, Optional

from .models import Post
from .providers import ExternalPostProvider
from ..utils.ingestion import ingest_posts

logger = structlog.get_logger()


class ThreadsToolkitProvider(ExternalPostProvider):
    """Fetch posts from a Threads Toolkit endpoint."""

    name = "threads_toolkit"

    def __init__(
        self,
        api_url: str,
        api_key: Optional[str] = None,
        query: Optional[str] = None,
        self_username: Optional[str] = None,
        max_age_hours: int = 24,
        max_items: int = 30,
        timeout: float = 15.0,
    ):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.query = query
        self.self_username = self_username
        self.max_age_hours = max_age_hours
        self.max_items = max_items
        self.timeout = timeout

    async def fetch_posts(self, max_items: int = 50) -> list[Post]:
        """Fetch posts; returns [] on any failure."""
        try:
            import httpx
        except Exception as exc:  # noqa: BLE001
            logger.warning("threads_toolkit_httpx_missing", error=str(exc))
            return []

        limit = min(max_items, self.max_items)
        params: dict[str, Any] = {"limit": limit}
        if self.query:
            params["query"] = self.query

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else None

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(self.api_url, params=params, headers=headers)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("threads_toolkit_fetch_failed", error=str(exc))
            return []

        # Allow API to return list or wrapped payload
        if isinstance(data, dict):
            items = data.get("items") or data.get("data") or data.get("results") or []
        else:
            items = data

        posts = ingest_posts(
            items,
            self_username=self.self_username,
            max_age_hours=self.max_age_hours,
        )
        logger.info("threads_toolkit_posts_fetched", count=len(posts))
        return posts
