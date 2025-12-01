"""Tests for the Persona Engine."""

import json
import tempfile
from pathlib import Path

import pytest

from src.agent.persona import (
    Identity,
    Interests,
    InteractionRules,
    Persona,
    Personality,
    SpeechPatterns,
)


class TestPersona:
    """Tests for Persona class."""

    def test_create_minimal_persona(self):
        """Test creating a persona with minimal required fields."""
        persona = Persona(
            identity=Identity(name="Test Agent"),
        )

        assert persona.identity.name == "Test Agent"
        assert persona.personality.communication_style == "casual and friendly"

    def test_create_full_persona(self):
        """Test creating a persona with all fields."""
        persona = Persona(
            identity=Identity(
                name="小光",
                age=28,
                occupation="設計師",
                location="台北",
                background="一個熱愛設計的人",
            ),
            personality=Personality(
                traits=["好奇", "幽默"],
                values=["真實", "創意"],
                communication_style="輕鬆自然",
            ),
            speech_patterns=SpeechPatterns(
                vocabulary_level="moderate",
                emoji_usage="occasional",
                typical_phrases=["有意思", "說真的"],
            ),
            interests=Interests(
                primary=["設計", "科技"],
                secondary=["咖啡", "攝影"],
                dislikes=["假掰"],
            ),
            interaction_rules=InteractionRules(
                max_response_length=280,
            ),
        )

        assert persona.identity.name == "小光"
        assert persona.identity.age == 28
        assert "好奇" in persona.personality.traits
        assert persona.speech_patterns.emoji_usage == "occasional"
        assert "設計" in persona.interests.primary

    def test_persona_from_file(self, tmp_path: Path):
        """Test loading persona from JSON file."""
        persona_data = {
            "identity": {
                "name": "Test Agent",
                "age": 25,
            },
            "personality": {
                "traits": ["curious"],
            },
        }

        file_path = tmp_path / "test_persona.json"
        with open(file_path, "w") as f:
            json.dump(persona_data, f)

        persona = Persona.from_file(file_path)

        assert persona.identity.name == "Test Agent"
        assert persona.identity.age == 25
        assert "curious" in persona.personality.traits

    def test_persona_to_file(self, tmp_path: Path):
        """Test saving persona to JSON file."""
        persona = Persona(
            identity=Identity(name="Save Test", age=30),
        )

        file_path = tmp_path / "saved_persona.json"
        persona.to_file(file_path)

        # Verify file was created and can be loaded
        with open(file_path, "r") as f:
            data = json.load(f)

        assert data["identity"]["name"] == "Save Test"
        assert data["identity"]["age"] == 30

    def test_get_system_prompt(self):
        """Test generating system prompt from persona."""
        persona = Persona(
            identity=Identity(
                name="小光",
                age=28,
                occupation="設計師",
                background="熱愛設計",
            ),
            personality=Personality(
                traits=["好奇", "幽默"],
                values=["真實"],
                communication_style="輕鬆自然",
            ),
        )

        prompt = persona.get_system_prompt()

        assert "小光" in prompt
        assert "28" in prompt
        assert "設計師" in prompt
        assert "好奇" in prompt
        assert "幽默" in prompt
        assert "輕鬆自然" in prompt

    def test_get_short_description(self):
        """Test getting short description."""
        persona = Persona(
            identity=Identity(name="小光"),
            personality=Personality(
                traits=["好奇", "幽默", "思考型", "友善"],
            ),
        )

        desc = persona.get_short_description()

        assert "小光" in desc
        # Should include first 3 traits
        assert "好奇" in desc
        assert "幽默" in desc
        assert "思考型" in desc


class TestInteractionRules:
    """Tests for InteractionRules."""

    def test_default_values(self):
        """Test default interaction rules."""
        rules = InteractionRules()

        assert "questions" in rules.respond_to
        assert "spam" in rules.avoid_responding_to
        assert rules.max_response_length == 280

    def test_custom_rules(self):
        """Test custom interaction rules."""
        rules = InteractionRules(
            respond_to=["tech discussions"],
            avoid_responding_to=["politics"],
            max_response_length=500,
        )

        assert "tech discussions" in rules.respond_to
        assert "politics" in rules.avoid_responding_to
        assert rules.max_response_length == 500
