"""Tests for Persona edge cases."""

import pytest

from src.agent.persona import Identity, Interests, Persona, Personality, SpeechPatterns


class TestPersonaEdgeCases:
    """Test edge cases for Persona class."""

    def test_persona_no_primary_interests(self):
        """Persona with empty primary interests should not crash."""
        persona = Persona(
            identity=Identity(name="Test", background="A test persona"),
            interests=Interests(primary=[], secondary=[]),
        )

        # Should generate system prompt without crashing
        prompt = persona.get_system_prompt()
        assert "Test" in prompt
        # Should handle empty interests gracefully
        assert "various topics" in prompt  # Default fallback

    def test_persona_no_traits(self):
        """Persona with empty traits should not crash."""
        persona = Persona(
            identity=Identity(name="Test", background="A test persona"),
            personality=Personality(traits=[], values=[]),
        )

        prompt = persona.get_system_prompt()
        assert "Test" in prompt
        # Should have fallback for empty traits
        assert "balanced" in prompt

    def test_persona_no_typical_phrases(self):
        """Persona with empty typical phrases should not crash."""
        persona = Persona(
            identity=Identity(name="Test", background="A test persona"),
            speech_patterns=SpeechPatterns(typical_phrases=[]),
        )

        prompt = persona.get_system_prompt()
        assert "Test" in prompt
        assert "none specific" in prompt

    def test_persona_minimal_config(self):
        """Persona with minimal config should work."""
        persona = Persona(
            identity=Identity(name="Minimal", background=""),
        )

        prompt = persona.get_system_prompt()
        assert "Minimal" in prompt

    def test_persona_emoji_usage_never(self):
        """Persona with emoji_usage=never should have clear instruction."""
        persona = Persona(
            identity=Identity(name="NoEmoji", background="Test"),
            speech_patterns=SpeechPatterns(emoji_usage="never"),
        )

        prompt = persona.get_system_prompt()
        assert "NEVER use emojis" in prompt

    def test_persona_emoji_usage_occasional(self):
        """Persona with emoji_usage=occasional should preserve setting."""
        persona = Persona(
            identity=Identity(name="SomeEmoji", background="Test"),
            speech_patterns=SpeechPatterns(emoji_usage="occasional"),
        )

        prompt = persona.get_system_prompt()
        assert "occasional" in prompt

    def test_get_short_description_with_few_traits(self):
        """Short description with less than 3 traits."""
        persona = Persona(
            identity=Identity(name="Test", background=""),
            personality=Personality(traits=["curious"]),
        )

        desc = persona.get_short_description()
        assert "Test" in desc
        assert "curious" in desc
