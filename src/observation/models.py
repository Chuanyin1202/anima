"""Data models for Observation Mode.

定義模擬過程中記錄的各種資料結構。
"""

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


def generate_id(prefix: str) -> str:
    """Generate a unique ID with prefix."""
    return f"{prefix}_{uuid4().hex[:8]}"


class RecordType(str, Enum):
    """Types of simulation records."""

    OBSERVATION = "observation"
    DECISION = "decision"
    RESPONSE = "response"
    REFLECTION = "reflection"


class PostData(BaseModel):
    """Serialized post data for logging."""

    id: str
    text: Optional[str] = None
    username: Optional[str] = None
    timestamp: Optional[datetime] = None
    media_type: Optional[str] = None
    permalink: Optional[str] = None


class ObservationRecord(BaseModel):
    """Record of a post observation."""

    id: str = Field(default_factory=lambda: generate_id("obs"))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    record_type: Literal["observation"] = "observation"
    post: PostData


class DecisionRecord(BaseModel):
    """Record of an engagement decision."""

    id: str = Field(default_factory=lambda: generate_id("dec"))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    record_type: Literal["decision"] = "decision"

    observation_id: str
    post_id: str
    should_engage: bool
    reason: str
    relevance_score: Optional[float] = None


class ResponseRecord(BaseModel):
    """Record of a generated response (not actually posted in observation mode)."""

    id: str = Field(default_factory=lambda: generate_id("res"))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    record_type: Literal["response"] = "response"

    decision_id: str
    post_id: str
    original_post_text: str
    generated_response: str
    adherence_score: float
    memory_context_used: list[str] = Field(default_factory=list)
    was_posted: bool = False  # Always False in observation mode
    refinement_attempts: int = 0


class ReflectionRecord(BaseModel):
    """Record of a reflection generated during simulation."""

    id: str = Field(default_factory=lambda: generate_id("ref"))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    record_type: Literal["reflection"] = "reflection"

    reflection_type: Literal["daily", "interaction", "topic"] = "daily"
    content: str
    based_on_memories: list[str] = Field(default_factory=list)


class LabelType(str, Enum):
    """Types of labels for responses."""

    GOOD = "good"
    BAD = "bad"
    NEUTRAL = "neutral"


class LabelRecord(BaseModel):
    """Record of a human label for a generated response."""

    id: str = Field(default_factory=lambda: generate_id("lbl"))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    record_type: Literal["label"] = "label"

    response_id: str
    label: LabelType
    reason: Optional[str] = None
    suggested_fix: Optional[str] = None
    reviewer: str = "human"

    # 標註細節（用於分析）
    issues: list[str] = Field(default_factory=list)
    # 例如: ["語氣太正式", "缺乏個人風格", "回應太長"]


class SimulationSession(BaseModel):
    """Metadata for a simulation session."""

    id: str = Field(default_factory=lambda: generate_id("sim"))
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    persona_name: str
    persona_file: str
    cycles_completed: int = 0
    total_observations: int = 0
    total_decisions: int = 0
    total_responses: int = 0
    total_reflections: int = 0

    def to_summary(self) -> dict[str, Any]:
        """Get session summary."""
        return {
            "session_id": self.id,
            "persona": self.persona_name,
            "duration_minutes": (
                (self.ended_at - self.started_at).total_seconds() / 60
                if self.ended_at
                else None
            ),
            "cycles": self.cycles_completed,
            "observations": self.total_observations,
            "decisions": self.total_decisions,
            "responses": self.total_responses,
            "reflections": self.total_reflections,
        }


# Type alias for any record type
SimulationRecord = (
    ObservationRecord | DecisionRecord | ResponseRecord | ReflectionRecord | LabelRecord
)
