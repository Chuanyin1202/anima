"""Helpers to ingest external JSON posts into Anima's internal Post model."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from ..threads import Post


def _parse_timestamp(ts: str | None) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _clean_text(text: str | None) -> str:
    return (text or "").strip()


def _is_self_post(username: str | None, self_username: str | None) -> bool:
    return bool(username and self_username and username.lower() == self_username.lower())


def ingest_posts(
    raw_posts: list[dict[str, Any]],
    self_username: str | None = None,
    max_age_hours: Optional[int] = None,
) -> list[Post]:
    """Convert external JSON posts into internal Post objects.

    Args:
        raw_posts: List of posts from toolkit/Apify (dicts).
        self_username: Optional username to filter out self posts.
        max_age_hours: Optional age filter; discard older posts.

    Returns:
        List of validated Post instances.
    """
    parsed: list[Post] = []
    now = datetime.now(timezone.utc)

    for item in raw_posts:
        username = item.get("author", {}).get("username") or item.get("username")
        content = _clean_text(item.get("content"))
        if not content or not username:
            continue
        if _is_self_post(username, self_username):
            continue

        ts = _parse_timestamp(item.get("timestamp"))
        if max_age_hours is not None and ts:
            age_hours = (now - ts).total_seconds() / 3600
            if age_hours > max_age_hours:
                continue

        post_id = str(item.get("id") or item.get("post_id") or "")
        if not post_id:
            # derive from url if possible
            url = item.get("url", "")
            if "/post/" in url:
                post_id = url.rsplit("/post/", 1)[-1]

        parsed.append(
            Post(
                id=post_id,
                username=username,
                text=content,
                url=item.get("url"),
                timestamp=ts,
                likes=item.get("stats", {}).get("likes") or item.get("likes"),
                replies=item.get("stats", {}).get("replies") or item.get("replies"),
                reposts=item.get("stats", {}).get("reposts") or item.get("reposts"),
                source=item.get("source"),
                parent_id=item.get("parentId") or item.get("parent_id"),
                quoted_post=item.get("quotedPost") or item.get("quoted_post"),
                media=item.get("images") or item.get("media"),
            )
        )

    return parsed
