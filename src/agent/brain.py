"""Agent Brain - Core decision-making and coordination logic.

This is the central orchestrator that:
1. Fetches content from Threads
2. Decides what to engage with
3. Generates persona-consistent responses
4. Manages memory and reflections
5. Handles rate limiting
"""

import asyncio
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import structlog
from openai import AsyncOpenAI

from ..memory import AgentMemory, ReflectionEngine
from ..observation import SimulationLogger
from ..threads import Post, ThreadsClient
from ..threads.client import ThreadsAPIError
from .persona import EMOJI_PATTERN, Persona, PersonaEngine
from ..utils.config import is_reasoning_model
from ..utils.ideas import format_ideas_for_context, get_recent_ideas

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
        model: str = "gpt-5-mini",
        advanced_model: str = "gpt-5.1",
        max_completion_tokens: int = 500,
        reasoning_effort: str = "low",
        max_interactions_per_cycle: int = 5,
        min_relevance_score: float = 0.6,
        observation_mode: bool = False,
        simulation_logger: Optional[SimulationLogger] = None,
    ):
        self.persona = persona
        self.threads = threads_client
        self.memory = memory
        self.openai = openai_client
        self.model = model
        self.advanced_model = advanced_model
        self.max_completion_tokens = max_completion_tokens
        self.reasoning_effort = reasoning_effort

        # Observation mode configuration
        self.observation_mode = observation_mode
        self.simulation_logger = simulation_logger

        # Initialize engines
        self.persona_engine = PersonaEngine(
            persona=persona,
            openai_client=openai_client,
            model=model,
            advanced_model=advanced_model,
            max_completion_tokens=max_completion_tokens,
            reasoning_effort=reasoning_effort,
        )
        self.reflection_engine = ReflectionEngine(
            memory=memory,
            openai_client=openai_client,
            persona_name=persona.identity.name,
            persona_description=persona.get_system_prompt(),
            model=model,
            max_completion_tokens=max_completion_tokens,
            reasoning_effort=reasoning_effort,
        )

        # Configuration
        self.max_interactions_per_cycle = max_interactions_per_cycle
        self.min_relevance_score = min_relevance_score

        # Tracking
        self._last_interaction_time: Optional[datetime] = None
        self._interactions_today = 0
        self._today_date = datetime.now(timezone.utc).date()
        self._self_username: Optional[str] = None

        if observation_mode:
            logger.info("observation_mode_enabled")

    @staticmethod
    def _is_simple_reaction(text: str) -> bool:
        """Heuristic check for short/reactive posts (emoji/讚/好厲害等)."""
        if not text:
            return False
        stripped = text.strip()
        # Very short or only emojis/punctuation
        if len(stripped) <= 10:
            return True
        # Common reaction words
        reaction_phrases = ["讚", "讚讚", "讚讚讚", "好厲害", "好強", "感謝", "謝謝", "笑死", "哈哈"]
        if any(phrase in stripped for phrase in reaction_phrases):
            return True
        # Emoji-only or almost only emoji
        if EMOJI_PATTERN.search(stripped):
            without_emoji = EMOJI_PATTERN.sub("", stripped)
            if len(without_emoji.strip()) <= 2:
                return True
        return False

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

    async def run_cycle(self, external_posts: Optional[list[Post]] = None) -> list[InteractionResult]:
        """Run one complete interaction cycle.

        This method:
        1. Checks if reflection is needed
        2. Fetches new posts to observe
        3. Decides which to engage with
        4. Generates and posts responses
        5. Records everything to memory

        Args:
            external_posts: Optional list of posts supplied externally (e.g., webhook).
                If provided, skips fetching and uses these posts directly.

        Returns:
            List of interaction results
        """
        logger.info("starting_cycle", agent=self.persona.identity.name)

        await self._ensure_clients_ready()

        # Reset daily counter if new day
        today = datetime.now(timezone.utc).date()
        if today != self._today_date:
            self._interactions_today = 0
            self._today_date = today

        results = []
        skipped_count = 0
        adherence_scores: list[float] = []
        refine_count = 0
        skip_by_reason: dict[str, int] = {}

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
            if external_posts:
                logger.info("using_external_posts", count=len(external_posts))
                posts = list(external_posts)
            else:
                posts = await self._fetch_interesting_posts()
                logger.info("posts_fetched", count=len(posts))
            print(f"\n{'='*60}", flush=True)
            print(f"Fetched {len(posts)} posts", flush=True)
            print(f"{'='*60}", flush=True)

            # Step 4: Observe and potentially interact
            interaction_count = 0
            for post in posts:
                if interaction_count >= self.max_interactions_per_cycle:
                    break

                # Print post content first
                post_text = (post.text or "")[:150]
                print(f"\n[Post #{post.id[:8]}]", flush=True)
                print(f"   Author: @{post.username}", flush=True)
                print(f"   Content: {post_text}{'...' if len(post.text or '') > 150 else ''}", flush=True)

                # Check if should skip (self-post or already interacted)
                skip_reason = self._get_skip_reason(post)
                if skip_reason:
                    print(f"   -> Skip: {skip_reason}", flush=True)
                    continue

                # Decide if we should engage (includes content filtering)
                should_engage, reason = await self.persona_engine.should_engage(
                    post.text or ""
                )

                # Print decision
                decision_str = "[REPLY]" if should_engage else "[SKIP]"
                print(f"   Decision: {decision_str}", flush=True)
                print(f"   Reason: {reason}", flush=True)

                # Log decision if logger is available (simulation or real)
                if self.simulation_logger:
                    self.simulation_logger.log_decision(
                        post_id=post.id,
                        should_engage=should_engage,
                        reason=reason,
                    )

                if not should_engage:
                    # Record skip summary for audit (not for context retrieval)
                    self.memory.record_skipped(
                        content=(post.text or "")[:100],
                        post_id=post.id,
                        skip_reason=reason,
                    )
                    skipped_count += 1
                    skip_by_reason[reason] = skip_by_reason.get(reason, 0) + 1
                    logger.debug("skipping_post", post_id=post.id, reason=reason)
                    continue

                # Record observation only for posts we engage with
                self.memory.observe(
                    content=post.text or "",
                    post_id=post.id,
                    author=post.username,
                )

                # Log observation if logger is available
                if self.simulation_logger:
                    self.simulation_logger.log_observation(post)

                # Check if post is actionable (has valid Graph ID)
                # Threads Graph API ONLY accepts numeric IDs.
                # Posts from external scrapers (Apify) usually have alphanumeric shortcodes (e.g. "DR9...").
                # There is no public way to convert Shortcode -> Graph ID without specific permissions.
                # Therefore, we treat shortcode posts as "Read-Only" (Market Research) - consume for memory but DO NOT reply.
                if not post.id.isdigit():
                    logger.info("skipping_reply_readonly_post", post_id=post.id, reason="shortcode_id_not_supported")
                    print(f"   -> Skip: Read-only post (Shortcode ID)", flush=True)
                    # We count this as a "successful" observation but not an interaction
                    continue

                # Try to interact
                result, score, refinements = await self._interact_with_post(post)
                results.append(result)
                if score is not None:
                    adherence_scores.append(score)
                refine_count += refinements

                if result.success:
                    interaction_count += 1
                    self._interactions_today += 1

                    # Random delay between interactions (skip in observation mode)
                    if not self.observation_mode:
                        delay = random.uniform(30, 120)
                        logger.debug("waiting_between_interactions", delay=delay)
                        await asyncio.sleep(delay)

            successful = len([r for r in results if r.success])
            logger.info(
                "cycle_complete",
                interactions=successful,
                total_attempts=len(results),
            )
            self._record_cycle_metrics(
                successful=successful,
                attempts=len(results),
                skipped=skipped_count,
                adherence_scores=adherence_scores,
                refine_count=refine_count,
                skip_by_reason=skip_by_reason,
            )
            print(f"\n{'='*60}", flush=True)
            print(f"Cycle complete: {successful}/{len(results)} successful interactions", flush=True)
            print(f"{'='*60}\n", flush=True)

        except Exception as e:
            logger.error("cycle_error", error=str(e))
            print(f"\nCycle error: {e}", flush=True)

        return results

    async def _fetch_interesting_posts(self) -> list[Post]:
        """Fetch posts to potentially interact with.

        Uses reply mode (replies to my posts) as primary source.
        Falls back to keyword search if available and no replies found.
        """
        posts = []

        # Primary: Get replies to my posts (no special permission needed)
        try:
            replies = await self.threads.get_replies_to_my_posts(
                max_posts=10,
                max_replies_per_post=5,
            )
            posts.extend(replies)
            logger.info("replies_mode", count=len(replies))
        except Exception as e:
            logger.warning("fetch_replies_failed", error=str(e))

        # Fallback: Search for posts (requires threads_keyword_search permission)
        if not posts:
            logger.info("fallback_to_search")
            for interest in self.persona.interests.primary[:3]:
                try:
                    search_result = await self.threads.search_posts(
                        query=interest,
                        limit=10,
                    )
                    posts.extend(search_result.posts)
                except Exception as e:
                    # Expected to fail without permission, just log debug
                    logger.debug("search_fallback_failed", interest=interest)

        # Deduplicate by post ID (preserve first occurrence)
        seen_ids = set()
        unique_posts = []
        for post in posts:
            if post.id in seen_ids:
                continue
            seen_ids.add(post.id)
            unique_posts.append(post)

        # Shuffle to add variety
        random.shuffle(unique_posts)

        return unique_posts[:20]  # Max 20 posts to consider

    async def _resolve_post_id(self, post: Post) -> Optional[str]:
        """Resolve a Threads Graph API friendly post ID.

        External sources may provide shortcodes/URLs; try to map them to an ID.
        """
        # Try as-is (may work if provider already supplies Graph ID)
        if post.id:
            try:
                await self.threads.get_post(post.id)
                return post.id
            except Exception:
                logger.debug("direct_id_failed", post_id=post.id, exc_info=True)

        # Numeric IDs are assumed valid
        if post.id and post.id.isdigit():
            return post.id

        # Try keyword search as a fallback (may require permission)
        if post.text:
            query = post.text[:64]
            try:
                search_result = await self.threads.search_posts(query=query, limit=5)
                for candidate in search_result.posts:
                    if post.username and candidate.username and candidate.username.lower() != post.username.lower():
                        continue
                    # Simple match: use first candidate that shares username
                    return candidate.id
            except Exception:
                logger.debug("search_resolve_failed", post_id=post.id, exc_info=True)

        return None

    async def _interact_with_post(self, post: Post) -> tuple[InteractionResult, Optional[float], int]:
        """Generate and post a response to a specific post.

        Returns:
            InteractionResult, adherence_score (or None if not computed), refinement_attempts
        """
        refinement_attempts = 0
        adherence_score: Optional[float] = None
        adherence_reason: Optional[str] = None
        response = ""  # Initialize to avoid NameError in except block

        try:
            # Get relevant memories for context
            participant_id = f"participant_{post.username}" if post.username else None
            is_reaction = self._is_simple_reaction(post.text or "")
            max_memories = 1 if is_reaction else 5
            memory_context = self.memory.get_context_for_response(
                post.text or "",
                max_memories=max_memories,
                min_relevance=0.7,
                participant_id=participant_id,
            )
            idea_context = format_ideas_for_context(
                get_recent_ideas(max_items=3, max_age_days=7, statuses=("pending", "posted"))
            )
            if idea_context and not is_reaction:
                memory_context = f"{memory_context}\n\n{idea_context}".strip()

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
            print(f"   Response: {response}", flush=True)

            # Verify persona adherence
            passes, score, adherence_reason = await self.persona_engine.verify_persona_adherence(response)
            adherence_score = score
            print(f"   Adherence: {score:.0%} ({'PASS' if passes else 'REFINE'})", flush=True)
            if adherence_reason:
                print(f"   Reason: {adherence_reason}", flush=True)

            if not passes:
                logger.info("refining_response", original_score=score, reason=adherence_reason)
                print(f"   Refining response...", flush=True)
                response = await self.persona_engine.refine_response(response)
                refinement_attempts += 1
                passes, score, adherence_reason = await self.persona_engine.verify_persona_adherence(
                    response
                )
                adherence_score = score
                print(f"   Refined: {response}", flush=True)
                print(f"   New adherence: {score:.0%}", flush=True)
                if adherence_reason:
                    print(f"   Reason: {adherence_reason}", flush=True)

                if not passes:
                    print(f"   [WARN] Still not matching persona, skipping", flush=True)
                    return (
                        InteractionResult(
                            success=False,
                            post_id=post.id,
                            reason="persona_adherence_failed",
                        ),
                        adherence_score,
                        refinement_attempts,
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
                        adherence_reason=adherence_reason,
                    )

                # Still record to memory for realistic simulation
                self.memory.record_interaction(
                    my_response=response,
                    context=post.text or "",
                    interaction_type="reply",
                    post_id=post.id,
                    participant_id=f"participant_{post.username}" if post.username else None,
                )

                logger.info(
                    "response_simulated",
                    post_id=post.id,
                    adherence_score=score,
                    response_preview=response[:50] + "..." if len(response) > 50 else response,
                )

                return (
                    InteractionResult(
                        success=True,
                        post_id=post.id,
                        response=response,
                        reason="simulated",
                    ),
                    adherence_score,
                    refinement_attempts,
                )

            # === NORMAL MODE: Actually post ===
            # Add signature if configured
            ai_signature = ""
            if self.persona.identity.signature:
                ai_signature = f"\n\n{self.persona.identity.signature}"
            response_with_signature = response + ai_signature

            # Resolve a Graph-compatible post ID (shortcode/url may fail)
            target_post_id = await self._resolve_post_id(post)
            if not target_post_id:
                logger.warning("reply_target_unresolved", post_id=post.id, username=post.username)
                return (
                    InteractionResult(success=False, post_id=post.id, reason="resolve_post_id_failed"),
                    adherence_score,
                    refinement_attempts,
                )

            async def _reply_with_retry(max_retries: int = 3) -> str:
                """Try replying with limited retries on transient errors."""
                delay = 1.0
                for attempt in range(1, max_retries + 1):
                    try:
                        return await self.threads.reply_to_post(target_post_id, response_with_signature)
                    except ThreadsAPIError as e:
                        transient = (e.status_code and e.status_code >= 500) or (e.error_code == 2)
                        if not transient or attempt == max_retries:
                            raise
                        logger.warning(
                            "reply_retrying_after_error",
                            attempt=attempt,
                            max_retries=max_retries,
                            post_id=target_post_id,
                            status_code=e.status_code,
                            error_code=e.error_code,
                            error=e.message,
                            delay=delay,
                        )
                        await asyncio.sleep(delay)
                        delay *= 2
                    except Exception as exc:  # noqa: BLE001
                        if attempt == max_retries:
                            raise
                        logger.warning(
                            "reply_retrying_after_unknown_error",
                            attempt=attempt,
                            max_retries=max_retries,
                            post_id=target_post_id,
                            error=str(exc),
                            delay=delay,
                        )
                        await asyncio.sleep(delay)
                        delay *= 2

            # Verify target post still exists before replying
            try:
                await self.threads.get_post(target_post_id)
            except ThreadsAPIError as e:
                if e.status_code == 404:
                    # Post was deleted or made private - skip this interaction
                    logger.warning(
                        "reply_target_deleted",
                        post_id=target_post_id,
                        username=post.username,
                    )
                    return (
                        InteractionResult(
                            success=False,
                            post_id=post.id,
                            reason="reply_target_deleted",
                        ),
                        adherence_score,
                        refinement_attempts,
                    )
                else:
                    # Re-raise other errors (5xx, network, etc.)
                    raise

            reply_id = await _reply_with_retry()

            # Log real posting result if logger is available
            if self.simulation_logger:
                self.simulation_logger.log_response(
                    post_id=post.id,
                    original_post_text=post.text or "",
                    generated_response=response,
                    adherence_score=adherence_score or score or 0,
                    memory_context_used=memory_lines,
                    refinement_attempts=refinement_attempts,
                    adherence_reason=adherence_reason,
                    was_posted=True,
                    error=None,
                )

            # Record the interaction in memory
            self.memory.record_interaction(
                my_response=response,
                context=post.text or "",
                interaction_type="reply",
                post_id=post.id,
                participant_id=f"participant_{post.username}" if post.username else None,
            )

            # Maybe do a quick reflection
            if random.random() < 0.3:  # 30% chance
                await self.reflection_engine.generate_interaction_reflection(
                    recent_interaction=response,
                    context=post.text or "",
                )

            self._last_interaction_time = datetime.now(timezone.utc)

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
            ), adherence_score, refinement_attempts

        except Exception as e:
            # Log posting failure if logger is available (real mode)
            if self.simulation_logger and not self.observation_mode:
                self.simulation_logger.log_response(
                    post_id=post.id,
                    original_post_text=post.text or "",
                    generated_response=response,
                    adherence_score=adherence_score or 0,
                    memory_context_used=[],
                    refinement_attempts=refinement_attempts,
                    adherence_reason=adherence_reason,
                    was_posted=False,
                    error=str(e),
                )

            logger.error("interaction_failed", post_id=post.id, error=str(e))
            return (
                InteractionResult(
                    success=False,
                    post_id=post.id,
                    reason=str(e),
                ),
                adherence_score,
                refinement_attempts,
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
            if self.persona.interests.primary:
                topic = random.choice(self.persona.interests.primary)
            elif self.persona.interests.secondary:
                topic = random.choice(self.persona.interests.secondary)
            else:
                logger.warning("no_interests_configured")
                return None

        # Get relevant memories + idea pool
        memory_context = self.memory.get_context_for_response(topic, max_memories=3)
        idea_context = format_ideas_for_context(
            get_recent_ideas(max_items=3, max_age_days=7, statuses=("pending", "posted"))
        )
        if idea_context:
            memory_context = f"{memory_context}\n\n{idea_context}".strip()

        # Generate the post
        prompt = f"""As {self.persona.identity.name}, write a short Threads post about: {topic}

{memory_context}

Guidelines:
- Be authentic to your personality
- Share a thought, observation, or question
- Keep it under {self.persona.interaction_rules.max_response_length} characters
- Don't be preachy or generic
"""

        kwargs = {
            "model": self.advanced_model,
            "messages": [
                {"role": "system", "content": self.persona.get_system_prompt()},
                {"role": "user", "content": prompt},
            ],
            "max_completion_tokens": self.max_completion_tokens,
        }
        if is_reasoning_model(self.advanced_model):
            kwargs["reasoning_effort"] = self.reasoning_effort

        response = await self.openai.chat.completions.create(**kwargs)

        post_content = response.choices[0].message.content or ""

        ai_signature = ""
        if self.persona.identity.signature:
            ai_signature = f"\n\n{self.persona.identity.signature}"

        # Enforce persona limit and Threads 500 char cap (safe hard stop)
        # Account for signature length
        max_len = min(self.persona.interaction_rules.max_response_length, 500) - len(ai_signature)
        if len(post_content) > max_len:
            post_content = post_content[: max_len - 3] + "..."

        # Keep content without signature for memory storage
        post_content_for_memory = post_content

        # Add signature for publishing
        post_content_with_signature = post_content + ai_signature

        try:
            post_id = await self.threads.create_post(post_content_with_signature)

            # Record in memory WITHOUT signature (no participant for original posts)
            self.memory.record_interaction(
                my_response=post_content_for_memory,
                context=f"Original post about {topic}",
                interaction_type="post",
                post_id=post_id,
                participant_id=None,  # Original post has no participant
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
        return self._get_skip_reason(post) is not None

    def _get_skip_reason(self, post: Post) -> str | None:
        """Get the reason for skipping a post, or None if should not skip."""
        if post.username and self._self_username and post.username == self._self_username:
            return "自己的貼文"

        if self.memory.has_interacted(post.id):
            return "已經互動過"

        return None

    async def _ensure_self_profile_cached(self) -> None:
        """Cache own username for self-reply avoidance."""
        if self._self_username:
            return
        try:
            profile = await self.threads.get_user_profile()
            self._self_username = profile.username
        except Exception:
            logger.debug("self_profile_fetch_failed", exc_info=True)

    def _record_cycle_metrics(
        self,
        successful: int,
        attempts: int,
        skipped: int,
        adherence_scores: list[float],
        refine_count: int,
        skip_by_reason: dict[str, int],
    ) -> None:
        """Persist simple cycle metrics to file for later analysis."""
        metrics_dir = Path("data/metrics")
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_file = metrics_dir / "cycle_metrics.jsonl"

        # Aggregate adherence stats
        avg_adherence = sum(adherence_scores) / len(adherence_scores) if adherence_scores else 0.0

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": self.persona.identity.name,
            "mode": "observe" if self.observation_mode else "normal",
            "successful": successful,
            "attempts": attempts,
            "skipped": skipped,
            "skip_by_reason": skip_by_reason,
            "adherence_avg": avg_adherence,
            "adherence_count": len(adherence_scores),
            "refine_count": refine_count,
            "interactions_today": self._interactions_today,
            "memory": self.memory.get_stats(),
        }

        try:
            with open(metrics_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            logger.warning("metrics_write_failed", file=str(metrics_file), exc_info=True)
