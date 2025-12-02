"""Mem0 integration adapter for agent memory.

This module provides a unified interface for storing and retrieving
agent memories using Mem0's intelligent memory layer.

Memory Types:
- Episodic: Specific interactions and events (posts seen, replies made)
- Semantic: Learned knowledge and facts from interactions
- Reflective: High-level insights generated through reflection
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

import structlog
from mem0 import Memory
from pydantic import BaseModel

logger = structlog.get_logger()


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
            # HTTPS: 用 host + port 443，讓 qdrant-client 自動偵測 HTTPS
            qdrant_config = {
                "host": parsed.netloc,
                "port": 443,
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
            "version": "v1.1",  # Enable graph memory features
        }

        # Add PostgreSQL for metadata if provided
        if database_url:
            config["history_db_path"] = database_url

        print("=== [5a] Calling Memory.from_config()... ===", flush=True)
        self.memory = Memory.from_config(config)
        print("=== [5b] Memory.from_config() completed ===", flush=True)
        logger.info("memory_initialized", agent_id=agent_id)

    def _format_metadata(
        self,
        memory_type: MemoryType,
        extra_metadata: Optional[dict] = None,
    ) -> dict:
        """Format metadata for a memory entry."""
        metadata = {
            "memory_type": memory_type.value,
            "timestamp": datetime.utcnow().isoformat(),
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
    ) -> str:
        """Record an interaction (e.g., replying to a post).

        Args:
            my_response: The response the agent gave
            context: The context/post being responded to
            interaction_type: Type of interaction (reply, like, etc.)
            post_id: The post ID being interacted with

        Returns:
            Memory ID
        """
        # Store as a conversation for better context extraction
        messages = [
            {"role": "user", "content": f"Context: {context}"},
            {"role": "assistant", "content": my_response},
        ]

        metadata = self._format_metadata(
            MemoryType.INTERACTION,
            {
                "interaction_type": interaction_type,
                "post_id": post_id,
            },
        )

        result = self.memory.add(
            messages=messages,
            user_id=self.agent_id,
            metadata=metadata,
        )

        memory_id = result.get("id", "unknown")
        logger.info(
            "interaction_recorded",
            memory_id=memory_id,
            interaction_type=interaction_type,
        )
        return memory_id

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
                    created_at=datetime.fromisoformat(
                        metadata.get("timestamp", datetime.utcnow().isoformat())
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
                    created_at=datetime.fromisoformat(
                        metadata.get("timestamp", datetime.utcnow().isoformat())
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
