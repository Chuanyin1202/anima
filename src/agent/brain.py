"""Agent Brain - Core decision-making and coordination logic.

This is the central orchestrator that:
1. Fetches content from Threads
2. Decides what to engage with
3. Generates persona-consistent responses
4. Manages memory and reflections
5. Handles rate limiting
"""

import asyncio
import random
from datetime import datetime, timedelta
from typing import Optional

import structlog
from openai import AsyncOpenAI

from ..memory import AgentMemory, ReflectionEngine
from ..observation import SimulationLogger
from ..threads import Post, ThreadsClient
from .persona import Persona, PersonaEngine

logger = structlog.get_logger()


class InteractionResult:
    """Result of an interaction attempt."""

    def __init__(
        self,
        success: bool,
        post_id: str,
        response: Optional[str] = None,
        reason: str = "",
    ):
        self.success = success
        self.post_id = post_id
        self.response = response
        self.reason = reason


class AgentBrain:
    """The brain of the agent - coordinates all decision-making.

    Usage:
        brain = AgentBrain(...)
        await brain.run_cycle()  # Run one interaction cycle
    """

    def __init__(
        self,
        persona: Persona,
        threads_client: ThreadsClient,
        memory: AgentMemory,
        openai_client: AsyncOpenAI,
        model: str = "gpt-4o-mini",
        advanced_model: str = "gpt-4o",
        max_interactions_per_cycle: int = 5,
        min_relevance_score: float = 0.6,
        observation_mode: bool = False,
        simulation_logger: Optional[SimulationLogger] = None,
    ):
        self.persona = persona
        self.threads = threads_client
        self.memory = memory
        self.openai = openai_client

        # Observation mode configuration
        self.observation_mode = observation_mode
        self.simulation_logger = simulation_logger

        # Initialize engines
        self.persona_engine = PersonaEngine(
            persona=persona,
            openai_client=openai_client,
            model=model,
            advanced_model=advanced_model,
        )
        self.reflection_engine = ReflectionEngine(
            memory=memory,
            openai_client=openai_client,
            persona_name=persona.identity.name,
            persona_description=persona.get_system_prompt(),
            model=model,
        )

        # Configuration
        self.max_interactions_per_cycle = max_interactions_per_cycle
        self.min_relevance_score = min_relevance_score

        # Tracking
        self._last_interaction_time: Optional[datetime] = None
        self._interactions_today = 0
        self._today_date = datetime.utcnow().date()
        self._self_username: Optional[str] = None

        if observation_mode:
            logger.info("observation_mode_enabled")

    async def __aenter__(self) -> "AgentBrain":
        await self._ensure_clients_ready()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def _ensure_clients_ready(self) -> None:
        """Ensure external clients are initialized before use."""
        await self.threads.open()
        await self._ensure_self_profile_cached()

    async def close(self) -> None:
        """Close any underlying resources."""
        try:
            await self.threads.close()
        except Exception:
            logger.debug("threads_close_failed", exc_info=True)
        try:
            await self.openai.aclose()
        except Exception:
            logger.debug("openai_close_failed", exc_info=True)

    async def run_cycle(self) -> list[InteractionResult]:
        """Run one complete interaction cycle.

        This method:
        1. Checks if reflection is needed
        2. Fetches new posts to observe
        3. Decides which to engage with
        4. Generates and posts responses
        5. Records everything to memory

        Returns:
            List of interaction results
        """
        logger.info("starting_cycle", agent=self.persona.identity.name)

        await self._ensure_clients_ready()

        # Reset daily counter if new day
        today = datetime.utcnow().date()
        if today != self._today_date:
            self._interactions_today = 0
            self._today_date = today

        results = []

        try:
            # Step 1: Maybe do reflection
            if await self.reflection_engine.should_reflect():
                logger.info("generating_reflection")
                await self.reflection_engine.generate_daily_reflection()

            # Step 2: Check rate limits
            if not await self.threads.can_reply():
                logger.warning("rate_limit_reached")
                return results

            # Step 3: Fetch posts to observe
            posts = await self._fetch_interesting_posts()
            logger.info("posts_fetched", count=len(posts))

            # Step 4: Observe and potentially interact
            interaction_count = 0
            for post in posts:
                if interaction_count >= self.max_interactions_per_cycle:
                    break

                if self._should_skip_post(post):
                    logger.debug("skipping_post_precheck", post_id=post.id)
                    continue

                # Record observation (to memory)
                self.memory.observe(
                    content=post.text or "",
                    post_id=post.id,
                    author=post.username,
                )

                # Log observation (for simulation)
                if self.observation_mode and self.simulation_logger:
                    self.simulation_logger.log_observation(post)

                # Decide if we should engage
                should_engage, reason = await self.persona_engine.should_engage(
                    post.text or ""
                )

                # Log decision (for simulation)
                if self.observation_mode and self.simulation_logger:
                    self.simulation_logger.log_decision(
                        post_id=post.id,
                        should_engage=should_engage,
                        reason=reason,
                    )

                if not should_engage:
                    logger.debug("skipping_post", post_id=post.id, reason=reason)
                    continue

                # Try to interact
                result = await self._interact_with_post(post)
                results.append(result)

                if result.success:
                    interaction_count += 1
                    self._interactions_today += 1

                    # Random delay between interactions (skip in observation mode)
                    if not self.observation_mode:
                        delay = random.uniform(30, 120)
                        logger.debug("waiting_between_interactions", delay=delay)
                        await asyncio.sleep(delay)

            logger.info(
                "cycle_complete",
                interactions=len([r for r in results if r.success]),
                total_attempts=len(results),
            )

        except Exception as e:
            logger.error("cycle_error", error=str(e))

        return results

    async def _fetch_interesting_posts(self) -> list[Post]:
        """Fetch posts that might be interesting to the agent."""
        posts = []

        # Search for posts related to interests
        for interest in self.persona.interests.primary[:3]:  # Top 3 interests
            try:
                search_result = await self.threads.search_posts(
                    query=interest,
                    limit=10,
                )
                posts.extend(search_result.posts)
            except Exception as e:
                logger.warning("search_failed", interest=interest, error=str(e))

        # Deduplicate by post ID
        seen_ids = set()
        unique_posts = []
        for post in posts:
            if post.id not in seen_ids:
                seen_ids.add(post.id)
                unique_posts.append(post)

        # Shuffle to add variety
        random.shuffle(unique_posts)

        return unique_posts[:20]  # Max 20 posts to consider

    async def _interact_with_post(self, post: Post) -> InteractionResult:
        """Generate and post a response to a specific post."""
        refinement_attempts = 0

        try:
            # Get relevant memories for context
            memory_context = self.memory.get_context_for_response(
                post.text or "",
                max_memories=5,
            )

            # Parse memory context for logging
            memory_lines = [
                line.strip("- ").strip()
                for line in memory_context.split("\n")
                if line.strip().startswith("-")
            ]

            # Generate response
            response = await self.persona_engine.generate_response(
                context=post.text or "",
                memory_context=memory_context,
            )

            # Verify persona adherence
            passes, score = await self.persona_engine.verify_persona_adherence(response)

            if not passes:
                logger.info("refining_response", original_score=score)
                response = await self.persona_engine.refine_response(response)
                refinement_attempts += 1
                passes, score = await self.persona_engine.verify_persona_adherence(
                    response
                )

                if not passes:
                    return InteractionResult(
                        success=False,
                        post_id=post.id,
                        reason="persona_adherence_failed",
                    )

            # === OBSERVATION MODE: Log but don't post ===
            if self.observation_mode:
                # Log the response (not actually posted)
                if self.simulation_logger:
                    self.simulation_logger.log_response(
                        post_id=post.id,
                        original_post_text=post.text or "",
                        generated_response=response,
                        adherence_score=score,
                        memory_context_used=memory_lines,
                        refinement_attempts=refinement_attempts,
                    )

                # Still record to memory for realistic simulation
                self.memory.record_interaction(
                    my_response=response,
                    context=post.text or "",
                    interaction_type="reply",
                    post_id=post.id,
                )

                logger.info(
                    "response_simulated",
                    post_id=post.id,
                    adherence_score=score,
                    response_preview=response[:50] + "..." if len(response) > 50 else response,
                )

                return InteractionResult(
                    success=True,
                    post_id=post.id,
                    response=response,
                    reason="simulated",
                )

            # === NORMAL MODE: Actually post ===
            reply_id = await self.threads.reply_to_post(post.id, response)

            # Record the interaction in memory
            self.memory.record_interaction(
                my_response=response,
                context=post.text or "",
                interaction_type="reply",
                post_id=post.id,
            )

            # Maybe do a quick reflection
            if random.random() < 0.3:  # 30% chance
                await self.reflection_engine.generate_interaction_reflection(
                    recent_interaction=response,
                    context=post.text or "",
                )

            self._last_interaction_time = datetime.utcnow()

            logger.info(
                "interaction_success",
                post_id=post.id,
                reply_id=reply_id,
                adherence_score=score,
            )

            return InteractionResult(
                success=True,
                post_id=post.id,
                response=response,
                reason="success",
            )

        except Exception as e:
            logger.error("interaction_failed", post_id=post.id, error=str(e))
            return InteractionResult(
                success=False,
                post_id=post.id,
                reason=str(e),
            )

    async def create_original_post(self, topic: Optional[str] = None) -> Optional[str]:
        """Create an original post (not a reply).

        Args:
            topic: Optional topic to post about. If None, chooses from interests.

        Returns:
            The post ID if successful, None otherwise.
        """
        if not await self.threads.can_publish():
            logger.warning("cannot_publish_rate_limit")
            return None

        await self._ensure_clients_ready()

        # Choose a topic if not provided
        if not topic:
            topic = random.choice(self.persona.interests.primary)

        # Get relevant memories
        memory_context = self.memory.get_context_for_response(topic, max_memories=3)

        # Generate the post
        prompt = f"""As {self.persona.identity.name}, write a short Threads post about: {topic}

{memory_context}

Guidelines:
- Be authentic to your personality
- Share a thought, observation, or question
- Keep it under {self.persona.interaction_rules.max_response_length} characters
- Don't be preachy or generic
"""

        response = await self.openai.chat.completions.create(
            model=self.persona_engine.model,
            messages=[
                {"role": "system", "content": self.persona.get_system_prompt()},
                {"role": "user", "content": prompt},
            ],
            max_tokens=150,
            temperature=0.9,
        )

        post_content = response.choices[0].message.content or ""

        # Enforce persona limit and Threads 500 char cap (safe hard stop)
        max_len = min(self.persona.interaction_rules.max_response_length, 500)
        if len(post_content) > max_len:
            post_content = post_content[: max_len - 3] + "..."

        try:
            post_id = await self.threads.create_post(post_content)

            # Record in memory
            self.memory.record_interaction(
                my_response=post_content,
                context=f"Original post about {topic}",
                interaction_type="post",
                post_id=post_id,
            )

            logger.info("original_post_created", post_id=post_id, topic=topic)
            return post_id

        except Exception as e:
            logger.error("original_post_failed", error=str(e))
            return None

    def get_stats(self) -> dict:
        """Get agent statistics."""
        memory_stats = self.memory.get_stats()

        return {
            "agent_name": self.persona.identity.name,
            "interactions_today": self._interactions_today,
            "last_interaction": self._last_interaction_time.isoformat()
            if self._last_interaction_time
            else None,
            "memory": memory_stats,
        }

    # =========================================================================
    # Helpers
    # =========================================================================

    def _should_skip_post(self, post: Post) -> bool:
        """Determine if a post should be skipped (self or already handled)."""
        if post.username and self._self_username and post.username == self._self_username:
            return True

        if self.memory.has_interacted(post.id):
            return True

        return False

    async def _ensure_self_profile_cached(self) -> None:
        """Cache own username for self-reply avoidance."""
        if self._self_username:
            return
        try:
            profile = await self.threads.get_user_profile()
            self._self_username = profile.username
        except Exception:
            logger.debug("self_profile_fetch_failed", exc_info=True)
