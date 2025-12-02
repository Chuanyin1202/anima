"""Reflection Engine - Generates high-level insights from memories.

Inspired by the reflection mechanism in Generative Agents (Park et al., 2023).
The agent periodically reflects on recent experiences to form higher-level
abstractions that guide future behavior.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

import structlog
from openai import AsyncOpenAI

from .mem0_adapter import AgentMemory, MemoryEntry, MemoryType

logger = structlog.get_logger()


class ReflectionEngine:
    """Generates reflections from agent memories.

    Reflections are higher-level insights that help maintain
    personality consistency and inform future interactions.
    """

    def __init__(
        self,
        memory: AgentMemory,
        openai_client: AsyncOpenAI,
        persona_name: str,
        persona_description: str,
        model: str = "gpt-4o-mini",
    ):
        self.memory = memory
        self.openai = openai_client
        self.persona_name = persona_name
        self.persona_description = persona_description
        self.model = model

    async def generate_daily_reflection(
        self,
        hours: int = 24,
        min_memories: int = 5,
    ) -> Optional[str]:
        """Generate a daily reflection based on recent experiences.

        Args:
            hours: Look back this many hours
            min_memories: Minimum memories required to generate reflection

        Returns:
            The generated reflection, or None if not enough memories
        """
        recent_memories = self.memory.get_recent(limit=50)

        # Filter to recent time window
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        relevant_memories = [m for m in recent_memories if m.created_at > cutoff]

        if len(relevant_memories) < min_memories:
            logger.info(
                "skipping_reflection",
                reason="not_enough_memories",
                found=len(relevant_memories),
                required=min_memories,
            )
            return None

        # Format memories for the prompt
        memories_text = self._format_memories(relevant_memories)

        reflection = await self._generate_reflection(
            memories_text,
            reflection_type="daily",
        )

        if reflection:
            # Store the reflection
            memory_ids = [m.id for m in relevant_memories]
            self.memory.add_reflection(reflection, based_on=memory_ids)

        return reflection

    async def generate_topic_reflection(
        self,
        topic: str,
        max_memories: int = 20,
    ) -> Optional[str]:
        """Generate a reflection focused on a specific topic.

        Args:
            topic: The topic to reflect on
            max_memories: Maximum memories to consider

        Returns:
            The generated reflection
        """
        memories = self.memory.search(topic, limit=max_memories)

        if not memories:
            return None

        memories_text = self._format_memories(memories)

        reflection = await self._generate_reflection(
            memories_text,
            reflection_type="topic",
            topic=topic,
        )

        if reflection:
            memory_ids = [m.id for m in memories]
            self.memory.add_reflection(reflection, based_on=memory_ids)

        return reflection

    async def generate_interaction_reflection(
        self,
        recent_interaction: str,
        context: str,
    ) -> Optional[str]:
        """Generate a quick reflection after an interaction.

        This is a lighter-weight reflection that happens after
        significant interactions to extract immediate insights.
        """
        prompt = f"""As {self.persona_name}, briefly reflect on this interaction:

Context: {context}
Your response: {recent_interaction}

In 1-2 sentences, note any important insight or observation from this exchange.
Consider: What did you learn? How does this fit with your existing views?
"""

        try:
            response = await self.openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.persona_description},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=150,
                temperature=0.7,
            )

            reflection = response.choices[0].message.content
            if reflection:
                self.memory.add_reflection(f"After interaction: {reflection}")
                logger.debug("interaction_reflection_generated")

            return reflection

        except Exception as e:
            logger.error("interaction_reflection_failed", error=str(e))
            return None

    def _format_memories(self, memories: list[MemoryEntry]) -> str:
        """Format memories into a readable text for the prompt."""
        lines = []
        for mem in memories:
            timestamp = mem.created_at.strftime("%Y-%m-%d %H:%M")
            lines.append(f"[{timestamp}] ({mem.memory_type.value}) {mem.content}")
        return "\n".join(lines)

    async def _generate_reflection(
        self,
        memories_text: str,
        reflection_type: str,
        topic: Optional[str] = None,
    ) -> Optional[str]:
        """Generate a reflection using LLM."""

        if reflection_type == "daily":
            focus = """
1. What patterns do you notice in your interactions?
2. What topics came up frequently?
3. What did you learn about the community?
4. Any thoughts or feelings about your day?
5. Are there topics you want to explore more?
"""
        elif reflection_type == "topic":
            focus = f"""
Focus specifically on: {topic}
1. What have you learned about this topic?
2. How have your views evolved?
3. What questions remain?
"""
        else:
            focus = "What insights can you draw from these experiences?"

        prompt = f"""As {self.persona_name}, reflect on your recent experiences:

{memories_text}

---

{focus}

Generate 3-5 high-level insights that capture the essence of these experiences.
Write in first person, as if you're journaling your thoughts.
Be specific but concise.
"""

        try:
            response = await self.openai.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": f"You are {self.persona_name}. {self.persona_description}",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=500,
                temperature=0.7,
            )

            reflection = response.choices[0].message.content
            logger.info(
                "reflection_generated",
                type=reflection_type,
                length=len(reflection) if reflection else 0,
            )
            return reflection

        except Exception as e:
            logger.error("reflection_generation_failed", error=str(e))
            return None

    async def should_reflect(self) -> bool:
        """Determine if it's time for a reflection.

        Returns True if:
        - No reflection in the last 12 hours
        - At least 10 new memories since last reflection
        """
        recent = self.memory.get_recent(limit=50)

        # Find last reflection
        reflections = [m for m in recent if m.memory_type == MemoryType.REFLECTIVE]

        if not reflections:
            # Never reflected, should if we have enough memories
            return len(recent) >= 10

        last_reflection = reflections[0]
        hours_since = (datetime.now(timezone.utc) - last_reflection.created_at).total_seconds() / 3600

        if hours_since < 12:
            return False

        # Count memories since last reflection
        new_memories = [m for m in recent if m.created_at > last_reflection.created_at]
        return len(new_memories) >= 10
