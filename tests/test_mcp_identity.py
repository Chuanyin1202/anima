"""Tests for MCP identity extraction."""

import pytest

from src.mcp.server import _extract_identity


class TestExtractIdentity:
    """Test identity extraction from messages."""

    def test_chinese_name_no_space(self):
        """Test Chinese name without space."""
        assert _extract_identity("我是小光") == "participant_小光"

    def test_chinese_name_with_space(self):
        """Test Chinese name with space."""
        assert _extract_identity("我是 小光") == "participant_小光"
        assert _extract_identity("我叫 Alex") == "participant_Alex"

    def test_english_name_im(self):
        """Test I'm pattern."""
        assert _extract_identity("I'm Alex") == "participant_Alex"
        assert _extract_identity("i'm Bob") == "participant_Bob"

    def test_english_name_my_name_is(self):
        """Test My name is pattern."""
        assert _extract_identity("My name is Bob") == "participant_Bob"
        assert _extract_identity("my name is alice") == "participant_alice"

    def test_english_name_this_is(self):
        """Test This is pattern."""
        assert _extract_identity("This is John") == "participant_John"
        assert _extract_identity("this is jane") == "participant_jane"

    def test_chinese_call_me(self):
        """Test 叫我 pattern."""
        assert _extract_identity("叫我小明") == "participant_小明"
        assert _extract_identity("改叫我大雄") == "participant_大雄"

    def test_ignore_url(self):
        """Test URL is not extracted as name."""
        assert _extract_identity("我是 https://example.com") is None
        assert _extract_identity("我是 http://test.com") is None
        assert _extract_identity("我是 www.example.com") is None

    def test_ignore_at_handle(self):
        """Test @ handles are not extracted as name."""
        assert _extract_identity("我是 @username") is None

    def test_no_match(self):
        """Test messages without identity declaration."""
        assert _extract_identity("你好") is None
        assert _extract_identity("今天天氣真好") is None
        assert _extract_identity("Hello there") is None

    def test_name_in_sentence(self):
        """Test identity extraction from longer sentences."""
        assert _extract_identity("嗨，我是小光，很高興認識你") == "participant_小光"
        assert _extract_identity("Hello, I'm Alex, nice to meet you") == "participant_Alex"

    def test_mixed_language_name(self):
        """Test mixed language names."""
        assert _extract_identity("我是Alex") == "participant_Alex"
        assert _extract_identity("I'm 小光") == "participant_小光"
