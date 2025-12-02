"""Mem0 integration adapter for agent memory.

This module provides a unified interface for storing and retrieving
agent memories using Mem0's intelligent memory layer.

Memory Types:
- Episodic: Specific interactions and events (posts seen, replies made)
- Semantic: Learned knowledge and facts from interactions
- Reflective: High-level insights generated through reflection
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import structlog
from mem0 import Memory
from pydantic import BaseModel

logger = structlog.get_logger()

# 自定義記憶萃取 prompt
# 角色區分由 mem0 的 agent_id vs user_id 機制自動處理
# 這裡只需要：繁體中文 + 防止腦補
CUSTOM_FACT_EXTRACTION_PROMPT = """
從對話中提取**實際出現**的重要資訊。

## 重要規則
1. **只記錄明確說出的內容** - 絕對不要推理、腦補或延伸
2. **使用繁體中文** - 所有記憶必須用繁體中文記錄
3. **寧缺勿濫** - 如果沒有值得記住的事實，返回空陣列

## 範例

Input: 你好
Output: {"facts": []}

Input: 我最近在研究 AI
Output: {"facts": ["最近在研究 AI"]}

Input: 我有個 side project 是做 AI 工具
Output: {"facts": ["有個 side project 是做 AI 工具"]}

Input: 在咖啡廳工作換個環境腦子就通了
Output: {"facts": ["認為在咖啡廳工作換環境腦子就通了"]}

Input: 我覺得創業最難的不是找資金，而是相信自己
Output: {"facts": ["認為創業最難的是相信自己，而不是找資金"]}

## 錯誤示範（絕對不要這樣做）
❌ "thinks AI is interesting" → 不要用英文
❌ "在咖啡廳工作錢包也空了" → 對話沒提到，不要腦補
❌ "可能對設計有興趣" → 不要推測，只記錄明確說出的

以 JSON 格式返回：{"facts": [...]}
"""


def parse_timestamp(ts: str) -> datetime:
    """Parse ISO timestamp and ensure timezone-aware.

    If the timestamp has no timezone info, assume UTC.
    """
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


class MemoryType(str, Enum):
    """Types of memories stored by the agent."""

    EPISODIC = "episodic"  # 情節記憶：看過的貼文、互動歷史
    SEMANTIC = "semantic"  # 語義記憶：學到的知識
    REFLECTIVE = "reflective"  # 反思記憶：高層次洞見
    OBSERVATION = "observation"  # 觀察：直接看到的內容
    INTERACTION = "interaction"  # 互動：回覆、按讚等行為


class MemoryEntry(BaseModel):
    """A single memory entry."""

    id: str
    content: str
    memory_type: MemoryType
    created_at: datetime
    metadata: dict[str, Any] = {}
    relevance_score: Optional[float] = None


class AgentMemory:
    """Agent memory system powered by Mem0.

    Provides a unified interface for:
    - Storing observations (posts seen)
    - Recording interactions (replies made)
    - Generating and storing reflections
    - Retrieving relevant memories for context
    """

    def __init__(
        self,
        agent_id: str,
        openai_api_key: str,
        qdrant_url: str = "http://localhost:6333",
        qdrant_api_key: Optional[str] = None,
        database_url: Optional[str] = None,
    ):
        self.agent_id = agent_id

        # Configure Mem0
        # Parse URL to handle HTTPS connections properly
        from urllib.parse import urlparse
        parsed = urlparse(qdrant_url)

        if parsed.scheme == "https":
            # HTTPS: 保留呼叫者指定的 port，未提供時才回退到 443
            qdrant_config = {
                "host": parsed.hostname or parsed.netloc,
                "port": parsed.port or 443,
                "collection_name": f"anima_{agent_id}",
            }
        else:
            # HTTP 可以直接用 url
            qdrant_config = {
                "url": qdrant_url,
                "collection_name": f"anima_{agent_id}",
            }

        if qdrant_api_key:
            qdrant_config["api_key"] = qdrant_api_key

        config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "gpt-4o-mini",
                    "api_key": openai_api_key,
                    "temperature": 0.1,  # 降低隨機性，減少腦補
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "text-embedding-3-small",
                    "api_key": openai_api_key,
                },
            },
            "vector_store": {
                "provider": "qdrant",
                "config": qdrant_config,
            },
            "custom_fact_extraction_prompt": CUSTOM_FACT_EXTRACTION_PROMPT,
            "version": "v1.1",  # Enable graph memory features
        }

        # Add PostgreSQL for metadata if provided
        if database_url:
            config["history_db_path"] = database_url

        self.memory = Memory.from_config(config)
        logger.info("memory_initialized", agent_id=agent_id)

    def _format_metadata(
        self,
        memory_type: MemoryType,
        extra_metadata: Optional[dict] = None,
    ) -> dict:
        """Format metadata for a memory entry."""
        metadata = {
            "memory_type": memory_type.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        return metadata

    # =========================================================================
    # Memory Storage
    # =========================================================================

    def observe(
        self,
        content: str,
        source: str = "threads",
        post_id: Optional[str] = None,
        author: Optional[str] = None,
    ) -> str:
        """Record an observation (e.g., seeing a post).

        Args:
            content: The content that was observed
            source: Source platform (default: threads)
            post_id: Optional post ID for reference
            author: Optional author username

        Returns:
            Memory ID
        """
        metadata = self._format_metadata(
            MemoryType.OBSERVATION,
            {
                "source": source,
                "post_id": post_id,
                "author": author,
            },
        )

        result = self.memory.add(
            messages=[{"role": "user", "content": content}],
            user_id=self.agent_id,
            metadata=metadata,
        )

        memory_id = result.get("id", "unknown")
        logger.debug("observation_recorded", memory_id=memory_id, source=source)
        return memory_id

    def record_interaction(
        self,
        my_response: str,
        context: str,
        interaction_type: str = "reply",
        post_id: Optional[str] = None,
        participant_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Record an interaction with proper role separation.

        Uses mem0's built-in role separation:
        - participant's content → stored with user_id (USER_MEMORY_EXTRACTION_PROMPT)
        - agent's response → stored with agent_id (AGENT_MEMORY_EXTRACTION_PROMPT)

        Args:
            my_response: The response the agent gave
            context: The context/post being responded to
            interaction_type: Type of interaction (reply, like, etc.)
            post_id: The post ID being interacted with
            participant_id: Identifier for the participant (e.g., "participant_Alex")

        Returns:
            Dict with participant_memory_id and agent_memory_id
        """
        # Default to unknown if no participant_id provided
        user_id = participant_id or "participant_unknown"

        metadata_base = {
            "interaction_type": interaction_type,
            "post_id": post_id,
            "participant_id": participant_id,
        }

        # 1. Record participant's content (uses USER_MEMORY_EXTRACTION_PROMPT)
        participant_metadata = self._format_metadata(
            MemoryType.INTERACTION,
            {**metadata_base, "about": "participant"},
        )
        participant_result = self.memory.add(
            messages=[{"role": "user", "content": context}],
            user_id=user_id,
            metadata=participant_metadata,
        )

        # 2. Record agent's response (uses AGENT_MEMORY_EXTRACTION_PROMPT)
        agent_metadata = self._format_metadata(
            MemoryType.INTERACTION,
            {**metadata_base, "about": "xiao_guang"},
        )
        agent_result = self.memory.add(
            messages=[{"role": "assistant", "content": my_response}],
            agent_id=self.agent_id,
            metadata=agent_metadata,
        )

        participant_memory_id = participant_result.get("id", "unknown")
        agent_memory_id = agent_result.get("id", "unknown")

        logger.info(
            "interaction_recorded",
            participant_memory_id=participant_memory_id,
            agent_memory_id=agent_memory_id,
            interaction_type=interaction_type,
            participant_id=participant_id,
        )

        return {
            "participant_memory_id": participant_memory_id,
            "agent_memory_id": agent_memory_id,
        }

    def add_reflection(self, insights: str, based_on: Optional[list[str]] = None) -> str:
        """Store a reflection (high-level insight).

        Args:
            insights: The reflection content
            based_on: List of memory IDs this reflection is based on

        Returns:
            Memory ID
        """
        metadata = self._format_metadata(
            MemoryType.REFLECTIVE,
            {"based_on_memories": based_on or []},
        )

        result = self.memory.add(
            messages=[{"role": "assistant", "content": f"Reflection: {insights}"}],
            user_id=self.agent_id,
            metadata=metadata,
        )

        memory_id = result.get("id", "unknown")
        logger.info("reflection_added", memory_id=memory_id)
        return memory_id

    # =========================================================================
    # Memory Retrieval
    # =========================================================================

    def search(
        self,
        query: str,
        limit: int = 10,
        memory_type: Optional[MemoryType] = None,
    ) -> list[MemoryEntry]:
        """Search for relevant memories.

        Args:
            query: Search query
            limit: Maximum number of results
            memory_type: Filter by memory type (optional)

        Returns:
            List of relevant memory entries
        """
        results = self.memory.search(
            query=query,
            user_id=self.agent_id,
            limit=limit,
        )

        entries = []
        for item in results.get("results", []):
            memory_data = item.get("memory", "")
            metadata = item.get("metadata", {})
            entry_type = MemoryType(metadata.get("memory_type", "observation"))

            # Filter by type if specified
            if memory_type and entry_type != memory_type:
                continue

            entries.append(
                MemoryEntry(
                    id=item.get("id", ""),
                    content=memory_data,
                    memory_type=entry_type,
                    created_at=parse_timestamp(
                        metadata.get("timestamp", datetime.now(timezone.utc).isoformat())
                    ),
                    metadata=metadata,
                    relevance_score=item.get("score"),
                )
            )

        logger.debug("memory_search", query=query[:50], results=len(entries))
        return entries

    def get_recent(
        self,
        limit: int = 20,
        memory_type: Optional[MemoryType] = None,
    ) -> list[MemoryEntry]:
        """Get recent memories.

        Args:
            limit: Maximum number of memories to retrieve
            memory_type: Filter by memory type (optional)

        Returns:
            List of recent memory entries
        """
        all_memories = self.memory.get_all(user_id=self.agent_id)

        entries = []
        for item in all_memories.get("results", []):
            memory_data = item.get("memory", "")
            metadata = item.get("metadata", {})
            entry_type = MemoryType(metadata.get("memory_type", "observation"))

            if memory_type and entry_type != memory_type:
                continue

            entries.append(
                MemoryEntry(
                    id=item.get("id", ""),
                    content=memory_data,
                    memory_type=entry_type,
                    created_at=parse_timestamp(
                        metadata.get("timestamp", datetime.now(timezone.utc).isoformat())
                    ),
                    metadata=metadata,
                )
            )

        # Sort by timestamp and limit
        entries.sort(key=lambda x: x.created_at, reverse=True)
        return entries[:limit]

    def get_context_for_response(
        self,
        post_content: str,
        max_memories: int = 5,
    ) -> str:
        """Get relevant context for generating a response.

        Args:
            post_content: The post we're responding to
            max_memories: Maximum number of memories to include

        Returns:
            Formatted context string
        """
        memories = self.search(post_content, limit=max_memories)

        if not memories:
            return ""

        context_parts = ["Relevant memories:"]
        for mem in memories:
            context_parts.append(f"- [{mem.memory_type.value}] {mem.content}")

        return "\n".join(context_parts)

    # =========================================================================
    # Memory Management
    # =========================================================================

    def delete(self, memory_id: str) -> bool:
        """Delete a specific memory."""
        try:
            self.memory.delete(memory_id)
            logger.info("memory_deleted", memory_id=memory_id)
            return True
        except Exception as e:
            logger.error("memory_delete_failed", memory_id=memory_id, error=str(e))
            return False

    def get_stats(self) -> dict:
        """Get memory statistics."""
        all_memories = self.memory.get_all(user_id=self.agent_id)
        memories_list = all_memories.get("results", [])

        stats = {
            "total_memories": len(memories_list),
            "by_type": {},
        }

        for item in memories_list:
            mem_type = item.get("metadata", {}).get("memory_type", "unknown")
            stats["by_type"][mem_type] = stats["by_type"].get(mem_type, 0) + 1

        return stats

    def has_interacted(self, post_id: str, search_limit: int = 200) -> bool:
        """Check whether the agent has already interacted with a given post."""
        recent = self.get_recent(limit=search_limit, memory_type=MemoryType.INTERACTION)
        return any(mem.metadata.get("post_id") == post_id for mem in recent)
