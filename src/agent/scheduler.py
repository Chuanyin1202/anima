"""Agent Scheduler - Manages periodic agent execution.

Handles:
- Periodic interaction cycles
- Daily reflection scheduling
- Rate limit aware scheduling
"""

import asyncio
import random
from datetime import datetime, time
from typing import Callable, Optional

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from .brain import AgentBrain

logger = structlog.get_logger()


class AgentScheduler:
    """Scheduler for running agent tasks periodically.

    Usage:
        scheduler = AgentScheduler(brain)
        scheduler.start()

        # Or run once:
        await scheduler.run_once()
    """

    def __init__(
        self,
        brain: AgentBrain,
        interaction_interval_hours: float = 4,
        reflection_time: time = time(hour=23, minute=0),  # 11 PM
        random_delay_minutes: int = 30,
    ):
        self.brain = brain
        self.interaction_interval_hours = interaction_interval_hours
        self.reflection_time = reflection_time
        self.random_delay_minutes = random_delay_minutes

        self._scheduler = AsyncIOScheduler()
        self._running = False

    def start(self) -> None:
        """Start the scheduler with all configured jobs."""
        if self._running:
            logger.warning("scheduler_already_running")
            return

        # Add interaction cycle job
        self._scheduler.add_job(
            self._run_interaction_cycle,
            IntervalTrigger(hours=self.interaction_interval_hours),
            id="interaction_cycle",
            name="Interaction Cycle",
            replace_existing=True,
            next_run_time=datetime.now(),  # Run immediately on start
        )

        # Add daily reflection job
        self._scheduler.add_job(
            self._run_daily_reflection,
            CronTrigger(
                hour=self.reflection_time.hour,
                minute=self.reflection_time.minute,
            ),
            id="daily_reflection",
            name="Daily Reflection",
            replace_existing=True,
        )

        self._scheduler.start()
        self._running = True

        logger.info(
            "scheduler_started",
            interaction_interval_hours=self.interaction_interval_hours,
            reflection_time=str(self.reflection_time),
        )

    def stop(self) -> None:
        """Stop the scheduler."""
        if not self._running:
            return

        self._scheduler.shutdown(wait=True)
        self._running = False
        logger.info("scheduler_stopped")

    async def run_once(self) -> dict:
        """Run a single interaction cycle (for manual/cron execution).

        Returns:
            Statistics about the run
        """
        logger.info("running_single_cycle")

        # Add random delay to appear more human
        delay = random.uniform(0, self.random_delay_minutes * 60)
        logger.debug("random_delay", seconds=delay)
        await asyncio.sleep(delay)

        results = await self.brain.run_cycle()

        stats = {
            "timestamp": datetime.utcnow().isoformat(),
            "successful_interactions": len([r for r in results if r.success]),
            "failed_interactions": len([r for r in results if not r.success]),
            "agent_stats": self.brain.get_stats(),
        }

        logger.info("single_cycle_complete", **stats)
        return stats

    async def _run_interaction_cycle(self) -> None:
        """Internal: Run interaction cycle with error handling."""
        try:
            # Add random delay
            delay = random.uniform(0, self.random_delay_minutes * 60)
            await asyncio.sleep(delay)

            await self.brain.run_cycle()

        except Exception as e:
            logger.error("interaction_cycle_error", error=str(e))

    async def _run_daily_reflection(self) -> None:
        """Internal: Run daily reflection."""
        try:
            reflection = await self.brain.reflection_engine.generate_daily_reflection()
            if reflection:
                logger.info("daily_reflection_complete", length=len(reflection))
        except Exception as e:
            logger.error("daily_reflection_error", error=str(e))


async def run_cli_mode(
    brain: AgentBrain,
    mode: str = "cycle",
    topic: Optional[str] = None,
) -> None:
    """Run the agent in CLI mode.

    Args:
        brain: The agent brain instance
        mode: "cycle" for interaction cycle, "post" for original post, "reflect" for reflection
        topic: Topic for original post (only used with mode="post")
    """
    if mode == "cycle":
        results = await brain.run_cycle()
        print(f"\nCompleted {len([r for r in results if r.success])} interactions")

    elif mode == "post":
        post_id = await brain.create_original_post(topic)
        if post_id:
            print(f"\nCreated post: {post_id}")
        else:
            print("\nFailed to create post")

    elif mode == "reflect":
        reflection = await brain.reflection_engine.generate_daily_reflection(
            min_memories=1  # Lower threshold for manual runs
        )
        if reflection:
            print(f"\nReflection:\n{reflection}")
        else:
            print("\nNo reflection generated (not enough memories)")

    elif mode == "stats":
        stats = brain.get_stats()
        import json
        print(json.dumps(stats, indent=2, ensure_ascii=False))

    else:
        print(f"Unknown mode: {mode}")
