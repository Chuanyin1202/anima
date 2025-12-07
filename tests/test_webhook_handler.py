import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

from src.webhooks.apify_webhook import ApifyWebhookHandler


@pytest.mark.asyncio
async def test_validate_and_filter_posts():
    handler = ApifyWebhookHandler(brain=MagicMock(), apify_api_token="token")

    recent_ts = datetime.now(timezone.utc).isoformat()
    old_ts = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()

    items = [
        {"id": "1", "text": "Valid", "timestamp": recent_ts},
        {"id": "2"},  # missing content
        "invalid",  # wrong type
        {"id": "3", "text": "Too old", "timestamp": old_ts},
    ]

    valid = handler._validate_and_filter_posts(items)

    assert len(valid) == 1
    assert valid[0]["id"] == "1"


@pytest.mark.asyncio
async def test_trigger_interaction_with_retry():
    brain = MagicMock()
    brain.run_cycle = AsyncMock(side_effect=[Exception("Temporary error"), None])

    handler = ApifyWebhookHandler(
        brain=brain,
        max_retries=3,
        retry_delay_base=0.01,
    )

    await handler._trigger_interaction([MagicMock()])

    assert brain.run_cycle.await_count == 2


@pytest.mark.asyncio
async def test_trigger_interaction_exhausted():
    brain = MagicMock()
    brain.run_cycle = AsyncMock(side_effect=Exception("Persistent error"))

    handler = ApifyWebhookHandler(
        brain=brain,
        max_retries=2,
        retry_delay_base=0.01,
    )

    await handler._trigger_interaction([MagicMock()])

    assert brain.run_cycle.await_count == 2
