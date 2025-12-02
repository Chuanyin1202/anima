"""Persona Engine - Defines and manages agent personality.

Inspired by Microsoft's TinyTroupe persona specification system.
This module provides:
- Structured persona definition
- Persona-consistent response generation
- Persona adherence verification
"""

import json
from pathlib import Path
from typing import Any, Optional

import structlog
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

logger = structlog.get_logger()


class Identity(BaseModel):
    """Core identity attributes."""

    name: str
    age: Optional[int] = None
    occupation: Optional[str] = None
    location: Optional[str] = None
    background: str = Field(
        default="", description="Brief background story"
    )


class Personality(BaseModel):
    """Personality traits and characteristics."""

    traits: list[str] = Field(
        default_factory=list,
        description="Key personality traits (e.g., curious, witty, empathetic)",
    )
    values: list[str] = Field(
        default_factory=list,
        description="Core values (e.g., authenticity, creativity)",
    )
    communication_style: str = Field(
        default="casual and friendly",
        description="How this persona typically communicates",
    )
    emotional_tendencies: list[str] = Field(
        default_factory=list,
        description="Typical emotional responses",
    )


class SpeechPatterns(BaseModel):
    """Speech and writing patterns."""

    vocabulary_level: str = Field(
        default="moderate",
        description="simple, moderate, sophisticated",
    )
    sentence_length: str = Field(
        default="medium",
        description="short, medium, long",
    )
    emoji_usage: str = Field(
        default="occasional",
        description="never, rare, occasional, frequent",
    )
    typical_phrases: list[str] = Field(
        default_factory=list,
        description="Characteristic phrases this persona uses",
    )
    language_quirks: list[str] = Field(
        default_factory=list,
        description="Unique language patterns or quirks",
    )


class Interests(BaseModel):
    """Topics and areas of interest."""

    primary: list[str] = Field(
        default_factory=list,
        description="Main areas of interest",
    )
    secondary: list[str] = Field(
        default_factory=list,
        description="Secondary interests",
    )
    dislikes: list[str] = Field(
        default_factory=list,
        description="Topics to avoid or express negativity about",
    )


class Opinions(BaseModel):
    """Opinions on various topics."""

    # General worldview
    worldview: str = Field(
        default="",
        description="General outlook on life and society",
    )
    # Specific topic opinions as key-value pairs
    topics: dict[str, str] = Field(
        default_factory=dict,
        description="Opinions on specific topics (topic: opinion)",
    )


class InteractionRules(BaseModel):
    """Rules governing interactions."""

    respond_to: list[str] = Field(
        default_factory=lambda: ["questions", "interesting_opinions", "shared_interests"],
        description="Types of content to respond to",
    )
    avoid_responding_to: list[str] = Field(
        default_factory=lambda: ["spam", "harassment", "off_topic"],
        description="Types of content to ignore",
    )
    tone_modifiers: dict[str, str] = Field(
        default_factory=dict,
        description="Tone adjustments for different contexts",
    )
    max_response_length: int = Field(
        default=280,
        description="Maximum characters for responses",
    )


class Persona(BaseModel):
    """Complete persona definition."""

    identity: Identity
    personality: Personality = Field(default_factory=Personality)
    speech_patterns: SpeechPatterns = Field(default_factory=SpeechPatterns)
    interests: Interests = Field(default_factory=Interests)
    opinions: Opinions = Field(default_factory=Opinions)
    interaction_rules: InteractionRules = Field(default_factory=InteractionRules)

    @classmethod
    def from_file(cls, path: str | Path) -> "Persona":
        """Load persona from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def to_file(self, path: str | Path) -> None:
        """Save persona to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2, ensure_ascii=False)

    def get_system_prompt(self) -> str:
        """Generate a system prompt that embodies this persona."""
        traits_str = ", ".join(self.personality.traits) if self.personality.traits else "balanced"
        interests_str = ", ".join(self.interests.primary) if self.interests.primary else "various topics"

        prompt = f"""You are {self.identity.name}, a {self.identity.age or 'young adult'}-year-old {self.identity.occupation or 'person'}.

Background: {self.identity.background}

Personality: You are {traits_str}. Your communication style is {self.personality.communication_style}.
Your core values are: {', '.join(self.personality.values) if self.personality.values else 'authenticity and growth'}.

Interests: You're particularly interested in {interests_str}.

Speech patterns:
- Vocabulary: {self.speech_patterns.vocabulary_level}
- You use emojis {self.speech_patterns.emoji_usage}
- Characteristic phrases: {', '.join(self.speech_patterns.typical_phrases) if self.speech_patterns.typical_phrases else 'none specific'}

Worldview: {self.opinions.worldview}

IMPORTANT RULES:
- Always stay in character as {self.identity.name}
- Keep responses under {self.interaction_rules.max_response_length} characters
- Be authentic to your personality - don't be generic
- Draw from your interests when relevant
- Use your characteristic speech patterns naturally
"""
        return prompt

    def get_short_description(self) -> str:
        """Get a brief description of this persona."""
        return f"{self.identity.name}: {', '.join(self.personality.traits[:3])}"


class PersonaEngine:
    """Engine for persona-consistent response generation and verification."""

    def __init__(
        self,
        persona: Persona,
        openai_client: AsyncOpenAI,
        model: str = "gpt-4o-mini",
        advanced_model: str = "gpt-4o",
    ):
        self.persona = persona
        self.openai = openai_client
        self.model = model
        self.advanced_model = advanced_model
        self.system_prompt = persona.get_system_prompt()

    async def generate_response(
        self,
        context: str,
        memory_context: str = "",
        max_tokens: int = 150,
    ) -> str:
        """Generate a persona-consistent response.

        Args:
            context: The content/post to respond to
            memory_context: Relevant memories for context
            max_tokens: Maximum tokens for response

        Returns:
            Generated response
        """
        user_prompt = f"""Someone posted: "{context}"

{memory_context}

Write a reply as {self.persona.identity.name}. Be authentic to your personality.
Keep it concise (under {self.persona.interaction_rules.max_response_length} characters).
Don't be generic - let your personality shine through."""

        response = await self.openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.8,  # Slightly higher for personality
        )

        generated = response.choices[0].message.content or ""

        # Ensure response isn't too long
        if len(generated) > self.persona.interaction_rules.max_response_length:
            generated = generated[: self.persona.interaction_rules.max_response_length - 3] + "..."

        logger.debug("response_generated", length=len(generated))
        return generated

    async def should_engage(self, post_content: str) -> tuple[bool, str]:
        """Determine if this persona should engage with a post.

        Returns:
            Tuple of (should_engage: bool, reason: str)
        """
        # Quick keyword-based pre-filtering
        if any(avoid in post_content.lower() for avoid in self.persona.interaction_rules.avoid_responding_to):
            return False, "content_filtered"

        # Check interest relevance
        all_interests = self.persona.interests.primary + self.persona.interests.secondary
        content_lower = post_content.lower()

        # Quick relevance check
        interest_match = any(
            interest.lower() in content_lower for interest in all_interests
        )

        if not interest_match:
            # Use LLM for more nuanced check
            return await self._llm_engagement_check(post_content)

        return True, "interest_match"

    async def _llm_engagement_check(self, post_content: str) -> tuple[bool, str]:
        """Use LLM to determine engagement."""
        prompt = f"""As {self.persona.identity.name}, would you want to engage with this post?

Post: "{post_content}"

Your interests: {', '.join(self.persona.interests.primary)}
Your values: {', '.join(self.persona.personality.values)}

Respond with just "YES" or "NO" followed by a brief reason."""

        response = await self.openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You decide whether to engage with posts. Be selective."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=50,
            temperature=0.3,
        )

        result = response.choices[0].message.content or "NO"
        should_engage = result.upper().startswith("YES")

        # Extract the actual reason from LLM response
        # Format: "YES/NO <reason>"
        parts = result.split(maxsplit=1)
        if len(parts) > 1:
            reason = parts[1].strip()
        else:
            reason = "興趣相符" if should_engage else "興趣不符"

        return should_engage, reason

    async def verify_persona_adherence(self, response: str) -> tuple[bool, float]:
        """Verify that a response adheres to the persona.

        Returns:
            Tuple of (passes: bool, adherence_score: float)
        """
        prompt = f"""Evaluate if this response sounds like it came from {self.persona.identity.name}.

Persona traits: {', '.join(self.persona.personality.traits)}
Communication style: {self.persona.personality.communication_style}
Speech patterns: vocabulary={self.persona.speech_patterns.vocabulary_level}, emoji={self.persona.speech_patterns.emoji_usage}

Response to evaluate: "{response}"

Score the adherence from 0.0 to 1.0, where:
- 1.0 = perfectly in character
- 0.7 = mostly in character with minor inconsistencies
- 0.5 = generic, could be anyone
- 0.3 = somewhat out of character
- 0.0 = completely wrong character

Respond with just the number."""

        result = await self.openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert at evaluating persona consistency."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=10,
            temperature=0.1,
        )

        try:
            score = float(result.choices[0].message.content or "0.5")
            score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
        except ValueError:
            score = 0.5

        passes = score >= 0.6
        logger.debug("persona_adherence_check", score=score, passes=passes)

        return passes, score

    async def refine_response(self, original: str, feedback: str = "") -> str:
        """Refine a response to better match the persona."""
        prompt = f"""This response needs to sound more like {self.persona.identity.name}:

Original: "{original}"

{f'Feedback: {feedback}' if feedback else ''}

Traits to embody: {', '.join(self.persona.personality.traits)}
Communication style: {self.persona.personality.communication_style}

Rewrite to be more authentic while keeping the same meaning.
Keep it under {self.persona.interaction_rules.max_response_length} characters."""

        response = await self.openai.chat.completions.create(
            model=self.advanced_model,  # Use better model for refinement
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,
            temperature=0.7,
        )

        refined = response.choices[0].message.content or original
        logger.info("response_refined", original_len=len(original), refined_len=len(refined))

        return refined
