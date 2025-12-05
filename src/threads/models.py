"""Data models for Threads API."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class MediaType(str, Enum):
    """Threads media types."""

    TEXT = "TEXT"
    TEXT_POST = "TEXT_POST"
    IMAGE = "IMAGE"
    VIDEO = "VIDEO"
    CAROUSEL = "CAROUSEL"
    CAROUSEL_ALBUM = "CAROUSEL_ALBUM"
    REPOST_FACADE = "REPOST_FACADE"
    AUDIO = "AUDIO"


class User(BaseModel):
    """Threads user profile."""

    id: str
    username: str
    name: Optional[str] = None
    threads_profile_picture_url: Optional[str] = None
    threads_biography: Optional[str] = None


class Post(BaseModel):
    """Threads post model."""

    id: str
    media_type: MediaType
    text: Optional[str] = None
    timestamp: datetime
    permalink: Optional[str] = None
    username: Optional[str] = None
    is_quote_post: bool = False
    shortcode: Optional[str] = None
    replied_to_id: Optional[str] = None  # Parent post ID if this is a reply

    # Search-specific fields
    has_replies: Optional[bool] = None
    is_reply: Optional[bool] = None

    # Engagement metrics (optional, from insights)
    likes: Optional[int] = None
    replies: Optional[int] = None
    reposts: Optional[int] = None
    quotes: Optional[int] = None
    views: Optional[int] = None


class Reply(BaseModel):
    """Threads reply model."""

    id: str
    text: str
    timestamp: datetime
    username: Optional[str] = None
    replied_to_id: str  # The post ID this is replying to


class PublishRequest(BaseModel):
    """Request model for publishing a post."""

    text: str = Field(..., max_length=500)
    media_type: MediaType = MediaType.TEXT
    reply_to_id: Optional[str] = None  # For replies


class RateLimitStatus(BaseModel):
    """Rate limit status from Threads API."""

    quota_usage: int
    quota_total: int
    reply_quota_usage: int = 0
    reply_quota_total: int = 1000


class SearchResult(BaseModel):
    """Search result from Threads API."""

    posts: list[Post]
    has_more: bool = False
    next_cursor: Optional[str] = None
