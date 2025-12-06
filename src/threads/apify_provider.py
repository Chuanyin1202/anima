"""Apify provider for fetching Threads posts via Apify Dataset.

Optional dependency; will degrade gracefully if httpx is not installed or
API calls fail (returns empty list).
"""

from __future__ import annotations

import structlog
from typing import Any, Optional

from .models import Post
from .providers import ExternalPostProvider
from ..utils.ingestion import ingest_posts

logger = structlog.get_logger()


class ApifyProvider(ExternalPostProvider):
    """Fetch posts from Apify Actor's latest dataset."""

    name = "apify"

    def __init__(
        self,
        api_token: str,
        actor_id: str,
        self_username: Optional[str] = None,
        max_age_hours: int = 24,
        max_items: int = 30,
        base_url: str = "https://api.apify.com/v2",
    ):
        self.api_token = api_token
        self.actor_id = actor_id
        self.self_username = self_username
        self.max_age_hours = max_age_hours
        self.max_items = max_items
        self.base_url = base_url.rstrip("/")

    async def fetch_posts(self, max_items: int = 50) -> list[Post]:
        """Fetch posts; returns [] on any failure."""
        try:
            raw_items = await self._fetch_dataset_items(limit=min(max_items, self.max_items))
            if not raw_items:
                return []

            posts = ingest_posts(
                raw_items,
                self_username=self.self_username,
                max_age_hours=self.max_age_hours,
            )
            logger.info("apify_posts_fetched", count=len(posts))
            return posts
        except Exception as exc:  # noqa: BLE001
            logger.warning("apify_fetch_failed", error=str(exc))
            return []

    async def _fetch_dataset_items(self, limit: int) -> list[dict[str, Any]]:
        """Fetch items from latest successful run's default dataset."""
        try:
            import httpx
        except Exception as exc:  # noqa: BLE001
            logger.warning("apify_httpx_missing", error=str(exc))
            return []

        # Step 1: latest run
        runs_url = f"{self.base_url}/acts/{self.actor_id}/runs"
        async with httpx.AsyncClient(timeout=15.0) as client:
            runs_resp = await client.get(
                runs_url,
                params={"token": self.api_token, "limit": 1, "desc": True},
            )
            runs_resp.raise_for_status()
            runs = runs_resp.json().get("data", {}).get("items", [])
            if not runs:
                return []

            dataset_id = runs[0].get("defaultDatasetId")
            if not dataset_id:
                return []

            # Step 2: dataset items
            items_url = f"{self.base_url}/datasets/{dataset_id}/items"
            items_resp = await client.get(
                items_url,
                params={"token": self.api_token, "limit": limit},
            )
            items_resp.raise_for_status()
            return items_resp.json()
