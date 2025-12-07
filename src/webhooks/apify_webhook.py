"""Apify webhook handler.

Receives webhook notifications from Apify when Actor runs complete,
processes the dataset items, and triggers interaction cycles.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Optional

import structlog

from ..utils.ingestion import ingest_posts

if TYPE_CHECKING:
    from ..agent.brain import AgentBrain

logger = structlog.get_logger()


class ApifyWebhookHandler:
    """Handler for Apify webhook notifications."""

    def __init__(
        self,
        brain: Optional["AgentBrain"] = None,
        self_username: Optional[str] = None,
        max_age_hours: int = 24,
        max_items: int = 30,
        apify_api_token: Optional[str] = None,
        base_url: str = "https://api.apify.com/v2",
        max_retries: int = 3,
        retry_delay_base: float = 2.0,
    ):
        """Initialize Apify webhook handler.

        Args:
            brain: AgentBrain instance to trigger interactions
            self_username: Username to filter out own posts
            max_age_hours: Maximum age of posts to process
            max_items: Maximum number of items to process
            apify_api_token: Apify API token for fetching datasets
            base_url: Apify API base URL
            max_retries: Maximum retry attempts when triggering interaction
            retry_delay_base: Base delay (seconds) for exponential backoff
        """
        self.brain = brain
        self.self_username = self_username
        self.max_age_hours = max_age_hours
        self.max_items = max_items
        self.apify_api_token = apify_api_token
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.retry_delay_base = retry_delay_base
        self._processing_lock = asyncio.Lock()

    def _validate_and_filter_posts(self, items: list[dict]) -> list[dict]:
        """Validate and filter dataset items before conversion."""
        valid_items: list[dict] = []

        for item in items:
            if not isinstance(item, dict):
                logger.warning("invalid_item_type", item_type=type(item))
                continue

            if not item.get("id"):
                logger.warning("missing_item_id", item=item)
                continue

            if not item.get("text") and not item.get("content"):
                logger.warning("missing_item_content", item_id=item.get("id"))
                continue

            timestamp = item.get("timestamp")
            if timestamp:
                try:
                    ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    cutoff = datetime.now(timezone.utc) - timedelta(hours=self.max_age_hours)
                    if ts < cutoff:
                        logger.debug("item_too_old", item_id=item.get("id"), timestamp=timestamp)
                        continue
                except Exception as exc:  # noqa: BLE001
                    logger.warning("invalid_timestamp", item_id=item.get("id"), error=str(exc))

            valid_items.append(item)

        logger.info("items_validated", total=len(items), valid=len(valid_items))
        return valid_items

    async def handle_webhook(self, payload: dict[str, Any]):
        """Handle Apify webhook notification.

        Expected payload structure:
        {
            "eventType": "ACTOR.RUN.SUCCEEDED",
            "resource": {
                "id": "run_id",
                "actId": "actor_id",
                "defaultDatasetId": "dataset_id"
            }
        }
        """
        # Prevent concurrent processing
        async with self._processing_lock:
            try:
                event_type = payload.get("eventType")
                if event_type != "ACTOR.RUN.SUCCEEDED":
                    logger.info(
                        "apify_webhook_ignored",
                        event_type=event_type,
                        reason="not_success_event",
                    )
                    return

                resource = payload.get("resource", {})
                dataset_id = resource.get("defaultDatasetId")
                run_id = resource.get("id")

                if not dataset_id:
                    logger.warning("apify_webhook_no_dataset", run_id=run_id)
                    return

                logger.info(
                    "apify_webhook_received",
                    run_id=run_id,
                    dataset_id=dataset_id,
                )

                # Fetch dataset items (this will be implemented in the provider)
                items = await self._fetch_dataset_items(dataset_id)
                if not items:
                    logger.info("apify_webhook_no_items", dataset_id=dataset_id)
                    return

                valid_items = self._validate_and_filter_posts(items)
                if not valid_items:
                    logger.warning("apify_webhook_no_valid_items", dataset_id=dataset_id)
                    return

                # Convert to Post objects
                posts = ingest_posts(
                    valid_items,
                    self_username=self.self_username,
                    max_age_hours=self.max_age_hours,
                )
                posts = posts[: self.max_items]

                logger.info(
                    "apify_webhook_posts_ingested",
                    total_items=len(items),
                    posts_count=len(posts),
                )

                # Trigger interaction if brain is available
                if self.brain and posts:
                    await self._trigger_interaction(posts)
                else:
                    logger.warning(
                        "apify_webhook_no_brain",
                        posts_count=len(posts),
                        reason="cannot_trigger_interaction",
                    )

            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "apify_webhook_handler_failed",
                    error=str(exc),
                    exc_info=True,
                )
                raise

    async def _fetch_dataset_items(self, dataset_id: str) -> list[dict[str, Any]]:
        """Fetch items from Apify dataset.

        Args:
            dataset_id: Apify dataset ID

        Returns:
            List of dataset items
        """
        if not self.apify_api_token:
            logger.error("apify_api_token_missing", dataset_id=dataset_id)
            return []

        try:
            import httpx
        except Exception as exc:  # noqa: BLE001
            logger.warning("apify_httpx_missing", error=str(exc))
            return []

        url = f"{self.base_url}/datasets/{dataset_id}/items"
        params = {
            "token": self.apify_api_token,
            "limit": self.max_items,
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                items = resp.json()

                logger.info(
                    "apify_dataset_fetched",
                    dataset_id=dataset_id,
                    items_count=len(items) if isinstance(items, list) else 0,
                )

                return items if isinstance(items, list) else []

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "apify_dataset_fetch_failed",
                dataset_id=dataset_id,
                error=str(exc),
                exc_info=True,
            )
            return []

    async def _trigger_interaction(self, posts: list):
        """Trigger brain to process posts with retry/backoff."""
        if not self.brain:
            logger.warning("no_brain_configured", posts_count=len(posts))
            return

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    "triggering_interaction_cycle",
                    posts_count=len(posts),
                    attempt=attempt,
                    max_retries=self.max_retries,
                )

                await self.brain.run_cycle(external_posts=posts)

                logger.info("interaction_cycle_completed", posts_count=len(posts))
                return
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "interaction_trigger_failed",
                    attempt=attempt,
                    max_retries=self.max_retries,
                    error=str(exc),
                    exc_info=True,
                )

                if attempt < self.max_retries:
                    import random

                    delay = (self.retry_delay_base**attempt) + random.uniform(0, 1)
                    logger.info("retrying_after_delay", delay_seconds=delay)
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "interaction_trigger_exhausted",
                        posts_count=len(posts),
                        total_attempts=self.max_retries,
                    )
