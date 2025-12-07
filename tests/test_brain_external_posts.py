import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from src.agent.brain import AgentBrain
from src.agent.persona import Persona
from src.threads.models import MediaType, Post


class StubMemory:
    def __init__(self):
        self.interactions = []
        self.observed = []
        self.skipped = []

    def record_skipped(self, **kwargs):
        self.skipped.append(kwargs)

    def observe(self, **kwargs):
        self.observed.append(kwargs)

    def record_interaction(self, **kwargs):
        self.interactions.append(kwargs)

    def get_context_for_response(self, *args, **kwargs):
        return "context"

    def has_interacted(self, post_id: str) -> bool:  # noqa: ARG002
        return False

    def get_stats(self):
        return {}


class StubPersonaEngine:
    def __init__(self):
        self.should_engage = AsyncMock(return_value=(True, "relevant"))
        self.generate_response = AsyncMock(return_value="response")
        self.verify_persona_adherence = AsyncMock(return_value=(True, 1.0, ""))
        self.refine_response = AsyncMock(return_value="refined")


class StubReflectionEngine:
    def __init__(self):
        self.should_reflect = AsyncMock(return_value=False)
        self.generate_daily_reflection = AsyncMock()
        self.generate_interaction_reflection = AsyncMock()


class StubThreadsClient:
    def __init__(self):
        self.can_reply = AsyncMock(return_value=True)

    async def open(self):
        return None

    async def close(self):
        return None


@pytest.mark.asyncio
async def test_run_cycle_with_external_posts():
    persona_path = Path(__file__).resolve().parents[1] / "personas" / "alex.json"
    persona = Persona.from_file(persona_path)

    brain = AgentBrain(
        persona=persona,
        threads_client=StubThreadsClient(),
        memory=StubMemory(),
        openai_client=MagicMock(aclose=AsyncMock()),
        observation_mode=True,
        simulation_logger=None,
        external_providers=[],
    )

    # Swap heavy components with stubs
    brain._ensure_clients_ready = AsyncMock()
    brain.reflection_engine = StubReflectionEngine()
    brain.persona_engine = StubPersonaEngine()
    brain._fetch_interesting_posts = AsyncMock()
    brain._record_cycle_metrics = MagicMock()

    external_posts = [
        Post(
            id="ext_1",
            media_type=MediaType.TEXT,
            text="External post for testing",
            timestamp=datetime.now(timezone.utc),
            username="someone_else",
        )
    ]

    results = await brain.run_cycle(external_posts=external_posts)

    assert len(results) == 1
    assert results[0].success
    brain._fetch_interesting_posts.assert_not_called()
    assert brain.persona_engine.generate_response.await_count == 1
    assert brain.memory.interactions
