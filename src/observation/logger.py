"""Simulation Logger for Observation Mode.

記錄模擬過程中的所有資料到 JSONL 檔案。
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import structlog

from ..threads import Post
from .models import (
    DecisionRecord,
    LabelRecord,
    ObservationRecord,
    PostData,
    ReflectionRecord,
    ResponseRecord,
    SimulationSession,
)

logger = structlog.get_logger()


class SimulationLogger:
    """Logger for recording simulation data to JSONL files.

    Data is stored in separate files for each record type:
    - observations.jsonl
    - decisions.jsonl
    - responses.jsonl
    - reflections.jsonl
    - labels.jsonl
    - sessions.jsonl
    """

    def __init__(self, output_dir: str | Path):
        """Initialize the simulation logger.

        Args:
            output_dir: Directory to store simulation data files.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self.observations_file = self.output_dir / "observations.jsonl"
        self.decisions_file = self.output_dir / "decisions.jsonl"
        self.responses_file = self.output_dir / "responses.jsonl"
        self.reflections_file = self.output_dir / "reflections.jsonl"
        self.labels_file = self.output_dir / "labels.jsonl"
        self.sessions_file = self.output_dir / "sessions.jsonl"

        # Current session tracking
        self._current_session: Optional[SimulationSession] = None

        # In-memory tracking for linking records
        self._last_observation_id: Optional[str] = None
        self._last_decision_id: Optional[str] = None

        logger.info("simulation_logger_initialized", output_dir=str(self.output_dir))

    def _append_to_file(self, filepath: Path, record: dict) -> None:
        """Append a record to a JSONL file."""
        with open(filepath, "a", encoding="utf-8") as f:
            # Handle datetime serialization
            json_str = json.dumps(record, ensure_ascii=False, default=str)
            f.write(json_str + "\n")

    def _read_all_records(self, filepath: Path) -> list[dict]:
        """Read all records from a JSONL file."""
        if not filepath.exists():
            return []

        records = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    # =========================================================================
    # Session Management
    # =========================================================================

    def start_session(self, persona_name: str, persona_file: str) -> SimulationSession:
        """Start a new simulation session."""
        self._current_session = SimulationSession(
            persona_name=persona_name,
            persona_file=persona_file,
        )
        logger.info("simulation_session_started", session_id=self._current_session.id)
        return self._current_session

    def end_session(self) -> Optional[SimulationSession]:
        """End the current session and save it."""
        if not self._current_session:
            return None

        self._current_session.ended_at = datetime.now(timezone.utc)
        self._append_to_file(
            self.sessions_file, self._current_session.model_dump()
        )

        session = self._current_session
        self._current_session = None
        logger.info("simulation_session_ended", session_id=session.id)
        return session

    def increment_cycle(self) -> None:
        """Increment the cycle counter for the current session."""
        if self._current_session:
            self._current_session.cycles_completed += 1

    # =========================================================================
    # Record Logging
    # =========================================================================

    def log_observation(self, post: Post) -> ObservationRecord:
        """Log an observed post.

        Args:
            post: The Threads post that was observed.

        Returns:
            The created observation record.
        """
        post_data = PostData(
            id=post.id,
            text=post.text,
            username=post.username,
            timestamp=post.timestamp,
            media_type=post.media_type.value if hasattr(post.media_type, 'value') else post.media_type,
            permalink=post.permalink,
        )

        record = ObservationRecord(post=post_data)
        self._append_to_file(self.observations_file, record.model_dump())

        # Track for linking
        self._last_observation_id = record.id

        # Update session stats
        if self._current_session:
            self._current_session.total_observations += 1

        logger.debug("observation_logged", record_id=record.id, post_id=post.id)
        return record

    def log_decision(
        self,
        post_id: str,
        should_engage: bool,
        reason: str,
        relevance_score: Optional[float] = None,
        observation_id: Optional[str] = None,
    ) -> DecisionRecord:
        """Log an engagement decision.

        Args:
            post_id: The post ID being decided on.
            should_engage: Whether to engage with the post.
            reason: The reason for the decision.
            relevance_score: Optional relevance score (0-1).
            observation_id: Link to the observation record.

        Returns:
            The created decision record.
        """
        record = DecisionRecord(
            observation_id=observation_id or self._last_observation_id or "",
            post_id=post_id,
            should_engage=should_engage,
            reason=reason,
            relevance_score=relevance_score,
        )
        self._append_to_file(self.decisions_file, record.model_dump())

        # Track for linking
        self._last_decision_id = record.id

        # Update session stats
        if self._current_session:
            self._current_session.total_decisions += 1

        logger.debug(
            "decision_logged",
            record_id=record.id,
            post_id=post_id,
            should_engage=should_engage,
        )
        return record

    def log_response(
        self,
        post_id: str,
        original_post_text: str,
        generated_response: str,
        adherence_score: float,
        memory_context_used: Optional[list[str]] = None,
        refinement_attempts: int = 0,
        decision_id: Optional[str] = None,
        adherence_reason: Optional[str] = None,
        was_posted: bool = False,
        error: Optional[str] = None,
    ) -> ResponseRecord:
        """Log a generated response (not actually posted).

        Args:
            post_id: The post being responded to.
            original_post_text: The original post content.
            generated_response: The generated response text.
            adherence_score: Persona adherence score (0-1).
            memory_context_used: List of memory content used for context.
            refinement_attempts: Number of times the response was refined.
            decision_id: Link to the decision record.
            adherence_reason: Reason for the adherence score from LLM.
            was_posted: Whether the response was actually sent (for real runs).
            error: Posting error message/code if not posted.

        Returns:
            The created response record.
        """
        record = ResponseRecord(
            decision_id=decision_id or self._last_decision_id or "",
            post_id=post_id,
            original_post_text=original_post_text,
            generated_response=generated_response,
            adherence_score=adherence_score,
            adherence_reason=adherence_reason,
            memory_context_used=memory_context_used or [],
            was_posted=was_posted,
            error=error,
            refinement_attempts=refinement_attempts,
        )
        self._append_to_file(self.responses_file, record.model_dump())

        # Update session stats
        if self._current_session:
            self._current_session.total_responses += 1

        logger.info(
            "response_logged",
            record_id=record.id,
            post_id=post_id,
            adherence_score=adherence_score,
        )
        return record

    def log_reflection(
        self,
        content: str,
        reflection_type: str = "daily",
        based_on_memories: Optional[list[str]] = None,
    ) -> ReflectionRecord:
        """Log a reflection.

        Args:
            content: The reflection content.
            reflection_type: Type of reflection (daily, interaction, topic).
            based_on_memories: List of memory IDs the reflection is based on.

        Returns:
            The created reflection record.
        """
        record = ReflectionRecord(
            reflection_type=reflection_type,  # type: ignore
            content=content,
            based_on_memories=based_on_memories or [],
        )
        self._append_to_file(self.reflections_file, record.model_dump())

        # Update session stats
        if self._current_session:
            self._current_session.total_reflections += 1

        logger.info("reflection_logged", record_id=record.id, type=reflection_type)
        return record

    def log_label(
        self,
        response_id: str,
        label: str,
        reason: Optional[str] = None,
        suggested_fix: Optional[str] = None,
        issues: Optional[list[str]] = None,
        reviewer: str = "human",
    ) -> LabelRecord:
        """Log a human label for a response.

        Args:
            response_id: The response being labeled.
            label: The label (good, bad, neutral).
            reason: Optional reason for the label.
            suggested_fix: Optional suggestion for improvement.
            issues: List of specific issues identified.
            reviewer: Who made the label.

        Returns:
            The created label record.
        """
        record = LabelRecord(
            response_id=response_id,
            label=label,  # type: ignore
            reason=reason,
            suggested_fix=suggested_fix,
            issues=issues or [],
            reviewer=reviewer,
        )
        self._append_to_file(self.labels_file, record.model_dump())

        logger.info(
            "label_logged", record_id=record.id, response_id=response_id, label=label
        )
        return record

    # =========================================================================
    # Data Retrieval
    # =========================================================================

    def get_observations(self) -> list[dict]:
        """Get all observation records."""
        return self._read_all_records(self.observations_file)

    def get_decisions(self) -> list[dict]:
        """Get all decision records."""
        return self._read_all_records(self.decisions_file)

    def get_responses(self) -> list[dict]:
        """Get all response records."""
        return self._read_all_records(self.responses_file)

    def get_reflections(self) -> list[dict]:
        """Get all reflection records."""
        return self._read_all_records(self.reflections_file)

    def get_labels(self) -> list[dict]:
        """Get all label records."""
        return self._read_all_records(self.labels_file)

    def get_sessions(self) -> list[dict]:
        """Get all session records."""
        return self._read_all_records(self.sessions_file)

    def get_unlabeled_responses(self) -> list[dict]:
        """Get responses that haven't been labeled yet."""
        responses = self.get_responses()
        labels = self.get_labels()

        labeled_ids = {label["response_id"] for label in labels}
        return [r for r in responses if r["id"] not in labeled_ids]

    def get_response_with_label(self, response_id: str) -> Optional[tuple[dict, Optional[dict]]]:
        """Get a response and its label (if any).

        Returns:
            Tuple of (response, label) or None if response not found.
        """
        responses = self.get_responses()
        labels = self.get_labels()

        response = next((r for r in responses if r["id"] == response_id), None)
        if not response:
            return None

        label = next((l for l in labels if l["response_id"] == response_id), None)
        return (response, label)

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> dict:
        """Get simulation statistics."""
        responses = self.get_responses()
        labels = self.get_labels()
        decisions = self.get_decisions()

        # Label distribution
        label_counts = {"good": 0, "bad": 0, "neutral": 0}
        for label in labels:
            label_type = label.get("label", "neutral")
            if label_type in label_counts:
                label_counts[label_type] += 1

        # Decision distribution
        engage_count = sum(1 for d in decisions if d.get("should_engage", False))

        # Average adherence score
        adherence_scores = [r.get("adherence_score", 0) for r in responses]
        avg_adherence = sum(adherence_scores) / len(adherence_scores) if adherence_scores else 0

        return {
            "total_observations": len(self.get_observations()),
            "total_decisions": len(decisions),
            "total_responses": len(responses),
            "total_reflections": len(self.get_reflections()),
            "total_labels": len(labels),
            "unlabeled_count": len(responses) - len(labels),
            "engagement_rate": engage_count / len(decisions) if decisions else 0,
            "label_distribution": label_counts,
            "average_adherence_score": round(avg_adherence, 3),
        }
