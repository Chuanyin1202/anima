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
        llm_model: str = "gpt-5-mini",
    ):
        self.agent_id = agent_id
        self.llm_model = llm_model

        # Patch mem0 qdrant adapter to avoid upserting vector=None (causes PointStruct errors)
        self._patch_mem0_qdrant_update()

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
                    "model": self.llm_model,
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
            "custom_fact_extraction_prompt": CUSTOM_FACT_EXTRACTION_PROMPT,
            # Graph memory v1.1 enables relationship extraction between memories.
            # The "event: NONE" / vector=None issue is fixed by _patch_mem0_qdrant_update().
            "version": "v1.1",
        }

        # Add PostgreSQL for metadata if provided
        if database_url:
            config["history_db_path"] = database_url

        self.memory = Memory.from_config(config)
        self.dedup_threshold = 0.85  # 相似度閾值，超過視為重複
        logger.info("memory_initialized", agent_id=agent_id)

    @staticmethod
    def _patch_mem0_qdrant_update() -> None:
        """Monkey-patch mem0 qdrant adapter: use set_payload when vector is None."""
        try:
            from qdrant_client.models import PointStruct  # Local import to avoid hard dep when mocked
        except Exception as exc:  # noqa: BLE001
            logger.warning("qdrant_pointstruct_import_failed", error=str(exc))
            return

        try:
            from mem0.vector_stores.qdrant import Qdrant as Mem0Qdrant
        except Exception as exc:  # noqa: BLE001
            logger.warning("mem0_qdrant_patch_import_failed", error=str(exc))
            return

        if getattr(Mem0Qdrant, "_anima_patched", False):
            return

        def patched_update(self, vector_id, vector=None, payload=None):  # type: ignore[override]
            # If vector is None, only update payload (keep existing embeddings)
            if vector is None:
                if payload:
                    try:
                        self.client.set_payload(
                            collection_name=self.collection_name,
                            payload=payload,
                            points=[vector_id],
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("qdrant_set_payload_failed", error=str(exc))
                return

            # Otherwise, behave like original update
            try:
                point = PointStruct(id=vector_id, vector=vector, payload=payload)
                self.client.upsert(collection_name=self.collection_name, points=[point])
            except Exception as exc:  # noqa: BLE001
                logger.warning("qdrant_upsert_failed", error=str(exc))

        Mem0Qdrant.update = patched_update
        Mem0Qdrant._anima_patched = True

    def _safe_add(
        self,
        messages: list[dict[str, str]],
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        metadata: Optional[dict] = None,
        context: str = "",
    ) -> dict[str, Any]:
        """Add memory entry with guard to avoid noisy PointStruct errors."""
        try:
            return self.memory.add(
                messages=messages,
                user_id=user_id,
                agent_id=agent_id,
                metadata=metadata,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "memory_add_failed",
                context=context,
                error=str(exc),
            )
            return {}

    def _has_post_id(self, post_id: str, search_limit: int = 100) -> bool:
        """Check if a post_id already exists in memory (exact match).

        Used for observe() dedup to avoid semantic confusion with summaries.

        Args:
            post_id: The post ID to check
            search_limit: Max memories to scan

        Returns:
            True if post_id already recorded
        """
        if not post_id:
            return False

        try:
            # Check both agent and user scopes (use search_limit for efficiency)
            agent_memories = self.memory.get_all(agent_id=self.agent_id, limit=search_limit)
            user_memories = self.memory.get_all(user_id=self.agent_id, limit=search_limit)

            all_items = (
                agent_memories.get("results", []) +
                user_memories.get("results", [])
            )

            for item in all_items:
                if item.get("metadata", {}).get("post_id") == post_id:
                    return True

            return False

        except Exception as e:
            logger.warning("post_id_check_failed", post_id=post_id, error=str(e))
            return False  # On error, allow write (conservative)

    def _is_duplicate_semantic(
        self,
        content: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> bool:
        """Check if semantically similar content already exists.

        Used for agent responses and reflections where semantic dedup makes sense.

        Args:
            content: Content to check for duplicates
            user_id: Search in user's memory scope
            agent_id: Search in agent's memory scope
            threshold: Similarity threshold (default: self.dedup_threshold)

        Returns:
            True if duplicate found, False otherwise
        """
        threshold = threshold or self.dedup_threshold

        # Skip very short content (not worth deduping)
        if len(content.strip()) < 10:
            return False

        try:
            # Search for similar content
            if user_id:
                results = self.memory.search(query=content, user_id=user_id, limit=3)
            elif agent_id:
                results = self.memory.search(query=content, agent_id=agent_id, limit=3)
            else:
                return False

            # Check if any result exceeds threshold
            for item in results.get("results", []):
                score = item.get("score", 0)
                if score >= threshold:
                    existing_memory = item.get("memory", "")[:50]
                    logger.debug(
                        "semantic_duplicate_detected",
                        score=round(score, 3),
                        existing=existing_memory,
                        new_content=content[:50],
                    )
                    return True

            return False

        except Exception as e:
            logger.warning("semantic_dedup_check_failed", error=str(e))
            return False  # On error, allow write (conservative)

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
    ) -> Optional[str]:
        """Record an observation (e.g., seeing a post).

        Args:
            content: The content that was observed
            source: Source platform (default: threads)
            post_id: Optional post ID for reference
            author: Optional author username

        Returns:
            Memory ID, or None if duplicate
        """
        # Check for duplicates using post_id (exact match, not semantic)
        # This avoids confusion with participant_summary which shares the same scope
        if post_id and self._has_post_id(post_id):
            logger.debug("observation_skipped_duplicate_post_id", post_id=post_id)
            return None

        metadata = self._format_metadata(
            MemoryType.OBSERVATION,
            {
                "source": source,
                "post_id": post_id,
                "author": author,
            },
        )

        result = self._safe_add(
            messages=[{"role": "user", "content": content}],
            user_id=self.agent_id,
            metadata=metadata,
            context="observe",
        )

        memory_id = result.get("id")
        if not memory_id:
            logger.warning("observation_record_failed", source=source, post_id=post_id)
            return None

        logger.debug("observation_recorded", memory_id=memory_id, source=source)
        return memory_id

    def record_skipped(
        self,
        content: str,
        post_id: str,
        skip_reason: str,
    ) -> Optional[str]:
        """Record a skipped post summary (for audit, not for context retrieval).

        Only recorded to agent scope, not participant memory.
        Marked with skipped=True in metadata to exclude from search().

        Args:
            content: First 100 chars of the post content
            post_id: The post ID that was skipped
            skip_reason: Reason for skipping (from should_engage)

        Returns:
            Memory ID if successful, None otherwise
        """
        metadata = self._format_metadata(
            MemoryType.OBSERVATION,
            {
                "source": "threads_skipped",
                "post_id": post_id,
                "skip_reason": skip_reason,
                "skipped": True,  # Flag to exclude from search
            },
        )

        # Format content with skip reason prefix
        summary = f"[skipped: {skip_reason[:50]}] {content}"

        try:
            result = self._safe_add(
                messages=[{"role": "user", "content": summary}],
                agent_id=self.agent_id,
                metadata=metadata,
                context="record_skipped",
            )
            memory_id = result.get("id")
            if not memory_id:
                logger.warning("skipped_record_failed", post_id=post_id, skip_reason=skip_reason[:30])
                return None
            logger.debug(
                "skipped_recorded",
                memory_id=memory_id,
                post_id=post_id,
                skip_reason=skip_reason[:30],
            )
            return memory_id
        except Exception as e:
            logger.warning("record_skipped_failed", post_id=post_id, error=str(e))
            return None

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
            Dict with participant_memory_id and agent_memory_id (None if skipped due to duplicate)
        """
        # Default to unknown if no participant_id provided
        user_id = participant_id or "participant_unknown"

        metadata_base = {
            "interaction_type": interaction_type,
            "post_id": post_id,
            "participant_id": participant_id,
        }

        participant_memory_id: Optional[str] = None
        agent_memory_id: Optional[str] = None
        skipped_count = 0
        errors: list[str] = []

        # Basic validation: skip if no content or no meaningful text
        if not context and not my_response:
            logger.warning("interaction_skipped_no_content", post_id=post_id, participant=participant_id)
            return {
                "participant_memory_id": None,
                "agent_memory_id": None,
                "skipped_duplicates": 0,
                "errors": ["no_content"],
            }
        if not context.strip() and not my_response.strip():
            logger.warning("interaction_skipped_empty_text", post_id=post_id, participant=participant_id)
            return {
                "participant_memory_id": None,
                "agent_memory_id": None,
                "skipped_duplicates": 0,
                "errors": ["empty_text"],
            }

        # 1. Record participant's content (with semantic dedup)
        if not self._is_duplicate_semantic(context, user_id=user_id):
            try:
                participant_metadata = self._format_metadata(
                    MemoryType.INTERACTION,
                    {**metadata_base, "about": "participant"},
                )
                participant_result = self._safe_add(
                    messages=[{"role": "user", "content": context}],
                    user_id=user_id,
                    metadata=participant_metadata,
                    context="interaction_participant",
                )
                participant_memory_id = participant_result.get("id")
                if not participant_memory_id:
                    errors.append("participant_memory_add_failed: no_id")
                    logger.warning("participant_memory_add_failed_no_id", post_id=post_id)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"participant_memory_add_failed: {exc}")
                logger.warning("participant_memory_add_failed", error=str(exc))
        else:
            skipped_count += 1
            logger.debug("participant_memory_skipped_duplicate", participant=user_id)

        # 2. Record agent's response (with semantic dedup)
        if not self._is_duplicate_semantic(my_response, agent_id=self.agent_id):
            try:
                agent_metadata = self._format_metadata(
                    MemoryType.INTERACTION,
                    {**metadata_base, "about": "self"},
                )
                agent_result = self._safe_add(
                    messages=[{"role": "assistant", "content": my_response}],
                    agent_id=self.agent_id,
                    metadata=agent_metadata,
                    context="interaction_agent",
                )
                agent_memory_id = agent_result.get("id")
                if not agent_memory_id:
                    errors.append("agent_memory_add_failed: no_id")
                    logger.warning("agent_memory_add_failed_no_id", post_id=post_id)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"agent_memory_add_failed: {exc}")
                logger.warning("agent_memory_add_failed", error=str(exc))
        else:
            skipped_count += 1
            logger.debug("agent_memory_skipped_duplicate")

        # 3. Copy participant summary to agent scope (with semantic dedup)
        summary_content = context[:300] + "..." if len(context) > 300 else context
        summary_text = f"[{user_id}] {summary_content}"
        if not self._is_duplicate_semantic(summary_text, user_id=self.agent_id):
            try:
                summary_metadata = self._format_metadata(
                    MemoryType.INTERACTION,
                    {**metadata_base, "about": "participant_summary", "source_participant": user_id},
                )
                self._safe_add(
                    messages=[{"role": "user", "content": summary_text}],
                    user_id=self.agent_id,
                    metadata=summary_metadata,
                    context="interaction_summary",
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"participant_summary_add_failed: {exc}")
                logger.warning("participant_summary_add_failed", error=str(exc))
        else:
            skipped_count += 1

        # Adjust log level based on whether anything was written
        if participant_memory_id or agent_memory_id:
            logger.info(
                "interaction_recorded",
                participant_memory_id=participant_memory_id,
                agent_memory_id=agent_memory_id,
                interaction_type=interaction_type,
                participant_id=participant_id,
                skipped_duplicates=skipped_count,
                errors=errors,
            )
        else:
            logger.debug(
                "interaction_all_duplicates_skipped",
                interaction_type=interaction_type,
                participant_id=participant_id,
                errors=errors,
            )

        return {
            "participant_memory_id": participant_memory_id,
            "agent_memory_id": agent_memory_id,
            "skipped_duplicates": skipped_count,
            "errors": errors,
        }

    def add_reflection(self, insights: str, based_on: Optional[list[str]] = None) -> Optional[str]:
        """Store a reflection (high-level insight).

        Args:
            insights: The reflection content
            based_on: List of memory IDs this reflection is based on

        Returns:
            Memory ID, or None if duplicate
        """
        reflection_content = f"Reflection: {insights}"

        # Check for duplicate reflections (semantic)
        if self._is_duplicate_semantic(reflection_content, user_id=self.agent_id):
            logger.debug("reflection_skipped_duplicate", content=insights[:50])
            return None

        metadata = self._format_metadata(
            MemoryType.REFLECTIVE,
            {"based_on_memories": based_on or []},
        )

        result = self._safe_add(
            messages=[{"role": "assistant", "content": reflection_content}],
            user_id=self.agent_id,
            metadata=metadata,
            context="add_reflection",
        )

        memory_id = result.get("id")
        if not memory_id:
            logger.warning("reflection_add_failed_unknown_id")
            return None

        logger.info("reflection_added", memory_id=memory_id)
        return memory_id

    # =========================================================================
    # Memory Retrieval
    # =========================================================================

    def _parse_memory_item(
        self,
        item: dict,
        memory_type_filter: Optional[MemoryType] = None,
        include_skipped: bool = False,
    ) -> Optional[MemoryEntry]:
        """Parse a memory item from mem0 response into MemoryEntry.

        Args:
            item: Raw memory item from mem0
            memory_type_filter: Filter by memory type (optional)
            include_skipped: Whether to include skipped memories (default: False)

        Returns:
            MemoryEntry if valid, None if filtered out
        """
        memory_data = item.get("memory", "")
        metadata = item.get("metadata", {})
        entry_type = MemoryType(metadata.get("memory_type", "observation"))

        # Exclude skipped memories from normal retrieval
        if not include_skipped and metadata.get("skipped"):
            return None

        if memory_type_filter and entry_type != memory_type_filter:
            return None

        return MemoryEntry(
            id=item.get("id", ""),
            content=memory_data,
            memory_type=entry_type,
            created_at=parse_timestamp(
                metadata.get("timestamp", datetime.now(timezone.utc).isoformat())
            ),
            metadata=metadata,
            relevance_score=item.get("score"),
        )

    def search(
        self,
        query: str,
        limit: int = 10,
        memory_type: Optional[MemoryType] = None,
    ) -> list[MemoryEntry]:
        """Search for relevant memories.

        Queries both agent memories (agent responses) and user memories
        (observations, old interactions stored under user_id=agent_id).

        Args:
            query: Search query
            limit: Maximum number of results
            memory_type: Filter by memory type (optional)

        Returns:
            List of relevant memory entries
        """
        # Query agent memories (agent's responses stored with agent_id)
        agent_results = self.memory.search(
            query=query,
            agent_id=self.agent_id,
            limit=limit,
        )

        # Query user memories (observations, legacy interactions)
        user_results = self.memory.search(
            query=query,
            user_id=self.agent_id,
            limit=limit,
        )

        # Merge and dedupe by ID
        seen_ids: set[str] = set()
        entries: list[MemoryEntry] = []

        for item in agent_results.get("results", []) + user_results.get("results", []):
            item_id = item.get("id", "")
            if item_id in seen_ids:
                continue
            seen_ids.add(item_id)

            entry = self._parse_memory_item(item, memory_type)
            if entry:
                entries.append(entry)

        # Sort by relevance score (descending), take top limit
        entries.sort(key=lambda x: x.relevance_score or 0, reverse=True)
        entries = entries[:limit]

        logger.debug("memory_search", query=query[:50], results=len(entries))
        return entries

    def get_recent(
        self,
        limit: int = 20,
        memory_type: Optional[MemoryType] = None,
    ) -> list[MemoryEntry]:
        """Get recent memories.

        Fetches both agent memories and user memories, merges and sorts by time.

        Args:
            limit: Maximum number of memories to retrieve
            memory_type: Filter by memory type (optional)

        Returns:
            List of recent memory entries
        """
        # Get agent memories (agent's responses)
        agent_memories = self.memory.get_all(agent_id=self.agent_id, limit=limit * 2)

        # Get user memories (observations, legacy interactions)
        user_memories = self.memory.get_all(user_id=self.agent_id, limit=limit * 2)

        # Merge and dedupe
        seen_ids: set[str] = set()
        entries: list[MemoryEntry] = []

        for item in agent_memories.get("results", []) + user_memories.get("results", []):
            item_id = item.get("id", "")
            if item_id in seen_ids:
                continue
            seen_ids.add(item_id)

            entry = self._parse_memory_item(item, memory_type)
            if entry:
                entries.append(entry)

        # Sort by timestamp (newest first) and limit
        entries.sort(key=lambda x: x.created_at, reverse=True)
        return entries[:limit]

    def get_context_for_response(
        self,
        post_content: str,
        max_memories: int = 5,
        participant_id: Optional[str] = None,
        min_relevance: float = 0.7,
    ) -> str:
        """Get relevant context for generating a response.

        Args:
            post_content: The post we're responding to
            max_memories: Maximum number of memories to include
            participant_id: Optional participant ID to also search their memories

        Returns:
            Formatted context string
        """
        # Request more than needed to allow for merging
        memories = self.search(post_content, limit=max_memories + 3)

        # If participant_id provided, also search their memories
        if participant_id:
            participant_results = self.memory.search(
                query=post_content,
                user_id=participant_id,
                limit=3,
            )
            seen_ids = {m.id for m in memories}
            for item in participant_results.get("results", []):
                entry = self._parse_memory_item(item)
                if entry and entry.id not in seen_ids:
                    memories.append(entry)
                    seen_ids.add(entry.id)

        if not memories:
            return ""

        # Sort by relevance score (fallback to timestamp if no score)
        memories.sort(
            key=lambda x: (x.relevance_score or 0, x.created_at.timestamp()),
            reverse=True,
        )
        # Filter by relevance threshold
        memories = [m for m in memories if (m.relevance_score or 0) >= min_relevance][:max_memories]

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
        """Get memory statistics (merges agent and user memories)."""
        # Get both agent and user memories (Mem0 defaults to limit=100, increase it)
        agent_memories = self.memory.get_all(agent_id=self.agent_id, limit=5000)
        user_memories = self.memory.get_all(user_id=self.agent_id, limit=5000)

        # Dedupe by ID
        seen_ids: set[str] = set()
        unique_items: list[dict] = []

        for item in agent_memories.get("results", []) + user_memories.get("results", []):
            item_id = item.get("id", "")
            if item_id not in seen_ids:
                seen_ids.add(item_id)
                unique_items.append(item)

        # Count skipped records
        skipped_count = sum(
            1 for item in unique_items if item.get("metadata", {}).get("skipped", False)
        )

        stats = {
            "total_memories": len(unique_items),
            "skipped_records": skipped_count,
            "by_type": {},
        }

        for item in unique_items:
            mem_type = item.get("metadata", {}).get("memory_type", "unknown")
            stats["by_type"][mem_type] = stats["by_type"].get(mem_type, 0) + 1

        return stats

    def get_skipped_records(self, limit: int = 50) -> list[dict]:
        """Get skipped post records for audit purposes."""
        all_memories = self.memory.get_all(agent_id=self.agent_id, limit=1000)

        skipped = []
        for item in all_memories.get("results", []):
            metadata = item.get("metadata", {})
            if metadata.get("skipped"):
                skipped.append(
                    {
                        "post_id": metadata.get("post_id"),
                        "skip_reason": metadata.get("skip_reason"),
                        "content": (item.get("memory", "") or "")[:100],
                        "timestamp": metadata.get("timestamp"),
                    }
                )

        return skipped[:limit]

    def has_interacted(self, post_id: str, search_limit: int = 200) -> bool:
        """Check whether the agent has already interacted with a given post."""
        recent = self.get_recent(limit=search_limit, memory_type=MemoryType.INTERACTION)
        return any(mem.metadata.get("post_id") == post_id for mem in recent)
