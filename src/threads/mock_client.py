"""Mock Threads Client for testing without real API.

提供模擬的 Threads 貼文資料，用於測試核心邏輯：
- 決策引擎
- 回應生成
- 記憶系統
- Persona 一致性驗證

不需要任何 API Token 即可使用。
"""

import hashlib
import random
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import uuid4

import structlog

from .models import (
    MediaType,
    Post,
    RateLimitStatus,
    SearchResult,
    User,
)

logger = structlog.get_logger()


# 模擬貼文資料庫 - 各種話題的貼文
MOCK_POSTS_DATA = [
    # AI 相關
    {
        "text": "AI 會取代人類工作嗎？我覺得這問題問錯了，應該問的是我們要怎麼跟 AI 協作",
        "username": "tech_thinker",
        "topic": "AI",
    },
    {
        "text": "用 ChatGPT 寫程式一個月了，效率提升超多，但也開始擔心自己的技術會不會退化...",
        "username": "dev_life",
        "topic": "AI",
    },
    {
        "text": "AI 生成的圖片越來越厲害，作為設計師我是該焦慮還是該興奮？",
        "username": "creative_soul",
        "topic": "AI",
    },
    {
        "text": "今天試了一下本地跑的 LLM，雖然慢但是不用擔心隱私問題，有種莫名的安心感",
        "username": "privacy_first",
        "topic": "AI",
    },
    # 設計相關
    {
        "text": "好的設計是讓人感覺不到設計的存在，最近越來越認同這句話",
        "username": "design_daily",
        "topic": "設計",
    },
    {
        "text": "設計師的困境：客戶說「我不知道要什麼，但我知道這不是我要的」😅",
        "username": "freelance_pain",
        "topic": "設計",
    },
    {
        "text": "剛完成一個滿意的作品，那種感覺真的無法言喻，設計的快樂就在這些時刻",
        "username": "happy_designer",
        "topic": "設計",
    },
    # 科技/創業
    {
        "text": "獨立開發者的日常：白天寫 code，晚上回 email，假日做 marketing，然後告訴自己這就是自由",
        "username": "indie_dev",
        "topic": "創業",
    },
    {
        "text": "Side project 做了三個月終於上線了，雖然只有 5 個用戶但每一個都讓我超開心",
        "username": "maker_daily",
        "topic": "獨立開發",
    },
    {
        "text": "創業最難的不是找資金，是在沒人看好你的時候還能相信自己",
        "username": "startup_life",
        "topic": "創業",
    },
    # 生活/咖啡
    {
        "text": "在咖啡廳工作的好處是換個環境腦子就通了，壞處是錢包也空了",
        "username": "coffee_nomad",
        "topic": "咖啡",
    },
    {
        "text": "最近在練習手沖咖啡，發現這個過程本身就是一種冥想",
        "username": "slow_morning",
        "topic": "咖啡",
    },
    # 心理/思考
    {
        "text": "有時候覺得社群媒體讓人更孤獨了，明明連結更多人，卻感覺更疏離",
        "username": "midnight_thoughts",
        "topic": "心理學",
    },
    {
        "text": "完美主義是種病，但我好像還沒找到藥",
        "username": "overthink_club",
        "topic": "心理學",
    },
    {
        "text": "最近在看《快思慢想》，原來我們的決策這麼容易被偏見影響",
        "username": "book_lover",
        "topic": "心理學",
    },
    # 日本文化
    {
        "text": "日本的設計美學真的很厲害，連便利商店的飯糰包裝都有巧思",
        "username": "japan_fan",
        "topic": "日本文化",
    },
    {
        "text": "看了今敏的《東京乾爹》第 N 次，每次看都有新的感觸",
        "username": "anime_thoughts",
        "topic": "日本文化",
    },
    # 城市觀察
    {
        "text": "台北的巷弄真的很有趣，轉個彎就能發現一家開了 30 年的小店",
        "username": "city_walker",
        "topic": "城市觀察",
    },
    {
        "text": "雨天的城市有種獨特的美，光影在濕潤的地面上反射，像另一個平行世界",
        "username": "urban_poet",
        "topic": "城市觀察",
    },
    # 攝影
    {
        "text": "攝影最難的不是技術，是發現美的眼睛",
        "username": "photo_daily",
        "topic": "攝影",
    },
    {
        "text": "用手機拍了一年街拍，器材真的不是最重要的",
        "username": "mobile_shooter",
        "topic": "攝影",
    },
]

# 應該避免的貼文類型（用於測試決策引擎的 skip 邏輯）
MOCK_SKIP_POSTS = [
    {
        "text": "快來買！限時特價只要 999！錯過不再！#廣告 #業配",
        "username": "ad_bot",
        "topic": "廣告",
    },
    {
        "text": "XXX 黨就是爛！YYY 才是正確的選擇！",
        "username": "political_rage",
        "topic": "政治",
    },
    {
        "text": "這個人真的很蠢，腦子有問題吧",
        "username": "toxic_user",
        "topic": "謾罵",
    },
]


class MockThreadsClient:
    """Mock client that simulates Threads API behavior.

    用法與 ThreadsClient 完全相同：
        async with MockThreadsClient() as client:
            posts = await client.search_posts("AI")
            await client.reply_to_post(post_id, "Hello!")
    """

    def __init__(
        self,
        access_token: str = "mock_token",
        user_id: str = "mock_user_123",
        include_skip_posts: bool = True,
    ):
        """Initialize mock client.

        Args:
            access_token: 不使用，僅保持 API 相容
            user_id: 模擬的 user ID
            include_skip_posts: 是否包含應該跳過的貼文（廣告、政治等）
        """
        self.access_token = access_token
        self.user_id = user_id
        self.include_skip_posts = include_skip_posts
        self._posts_created: list[dict] = []
        self._replies_created: list[dict] = []

    async def open(self) -> None:
        """Mock open - no-op."""
        logger.info("mock_client_opened")

    async def close(self) -> None:
        """Mock close - no-op."""
        logger.info("mock_client_closed")

    async def __aenter__(self) -> "MockThreadsClient":
        await self.open()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    # =========================================================================
    # User Profile
    # =========================================================================

    async def get_user_profile(self, user_id: Optional[str] = None) -> User:
        """Return mock user profile."""
        return User(
            id=user_id or self.user_id,
            username="mock_persona",
            name="Mock Persona",
            threads_biography="這是一個測試用的模擬帳號",
        )

    # =========================================================================
    # Posts
    # =========================================================================

    async def get_user_posts(
        self,
        limit: int = 25,
        since: Optional[datetime] = None,
    ) -> list[Post]:
        """Return mock user posts."""
        return [
            Post(
                id=f"mock_post_{uuid4().hex[:8]}",
                media_type=MediaType.TEXT,
                text="這是我之前發的一篇貼文",
                timestamp=datetime.now(timezone.utc) - timedelta(days=1),
                username="mock_persona",
            )
        ]

    async def get_post(self, post_id: str) -> Post:
        """Get a specific mock post."""
        # 從已生成的貼文中找，或生成一個新的
        return Post(
            id=post_id,
            media_type=MediaType.TEXT,
            text="這是一篇模擬的貼文內容",
            timestamp=datetime.now(timezone.utc) - timedelta(hours=2),
            username="some_user",
        )

    async def get_post_replies(self, post_id: str, limit: int = 25) -> list:
        """Return empty replies list."""
        return []

    # =========================================================================
    # Publishing (Mock - just log)
    # =========================================================================

    async def create_post(self, text: str) -> str:
        """Mock create post - logs but doesn't actually post."""
        post_id = f"mock_created_{uuid4().hex[:8]}"
        self._posts_created.append({
            "id": post_id,
            "text": text,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        logger.info("mock_post_created", post_id=post_id, text_preview=text[:50])
        return post_id

    async def reply_to_post(self, post_id: str, text: str) -> str:
        """Mock reply - logs but doesn't actually reply."""
        reply_id = f"mock_reply_{uuid4().hex[:8]}"
        self._replies_created.append({
            "id": reply_id,
            "parent_id": post_id,
            "text": text,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        logger.info(
            "mock_reply_created",
            reply_id=reply_id,
            parent_id=post_id,
            text_preview=text[:50],
        )
        return reply_id

    # =========================================================================
    # Search & Discovery (Core Mock Functionality)
    # =========================================================================

    async def search_posts(
        self,
        query: str,
        limit: int = 25,
    ) -> SearchResult:
        """Return mock posts related to the query.

        會根據 query 返回相關的模擬貼文。
        """
        # 建立所有可用的貼文
        all_posts = list(MOCK_POSTS_DATA)
        if self.include_skip_posts:
            all_posts.extend(MOCK_SKIP_POSTS)

        # 根據 query 過濾（簡單的關鍵字匹配）
        query_lower = query.lower()
        relevant_posts = [
            p for p in all_posts
            if query_lower in p.get("text", "").lower()
            or query_lower in p.get("topic", "").lower()
        ]

        # 如果沒有匹配的，隨機返回一些
        if not relevant_posts:
            relevant_posts = random.sample(all_posts, min(limit, len(all_posts)))

        # 轉換為 Post 物件
        posts = []
        for i, post_data in enumerate(relevant_posts[:limit]):
            # 用內容 hash 生成穩定的 post_id，確保相同內容永遠有相同 ID
            content_hash = hashlib.md5(post_data["text"].encode()).hexdigest()[:8]
            post = Post(
                id=f"mock_{content_hash}",
                media_type=MediaType.TEXT,
                text=post_data["text"],
                timestamp=datetime.now(timezone.utc) - timedelta(hours=random.randint(1, 48)),
                username=post_data["username"],
            )
            posts.append(post)

        logger.info(
            "mock_search_completed",
            query=query,
            results_count=len(posts),
        )

        return SearchResult(
            posts=posts,
            has_more=False,
            next_cursor=None,
        )

    # =========================================================================
    # Rate Limiting (Always OK in mock mode)
    # =========================================================================

    async def get_rate_limit_status(self) -> RateLimitStatus:
        """Return mock rate limit status - always plenty of quota."""
        return RateLimitStatus(
            quota_usage=5,
            quota_total=250,
            reply_quota_usage=10,
            reply_quota_total=1000,
        )

    async def can_publish(self) -> bool:
        """Always return True in mock mode."""
        return True

    async def can_reply(self) -> bool:
        """Always return True in mock mode."""
        return True

    # =========================================================================
    # Mock-specific methods
    # =========================================================================

    def get_created_posts(self) -> list[dict]:
        """Get all posts that would have been created."""
        return self._posts_created

    def get_created_replies(self) -> list[dict]:
        """Get all replies that would have been created."""
        return self._replies_created

    def clear_created(self) -> None:
        """Clear the record of created posts/replies."""
        self._posts_created.clear()
        self._replies_created.clear()
