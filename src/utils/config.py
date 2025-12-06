"""Configuration management using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# 專案根目錄（無論從哪裡啟動都能找到 .env）
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # OpenAI
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_model: str = Field(default="gpt-5-mini", description="Default OpenAI model")
    openai_model_advanced: str = Field(
        default="gpt-5.1", description="Advanced model for complex tasks (posting)"
    )
    # gpt-5 系列需要足夠的 tokens 給 reasoning + output
    max_completion_tokens: int = Field(
        default=500, description="Max completion tokens for gpt-5 series"
    )
    reasoning_effort: str = Field(
        default="low", description="Reasoning effort for gpt-5 series"
    )

    # Threads API
    threads_app_id: str = Field(default="", description="Meta Threads App ID")
    threads_app_secret: str = Field(default="", description="Meta Threads App Secret")
    threads_access_token: str = Field(default="", description="Threads access token")
    threads_user_id: str = Field(default="", description="Threads user ID")
    threads_username: str = Field(default="", description="Threads username (for self-filter)")

    # Mock Mode (for testing without real API)
    use_mock_threads: bool = Field(
        default=False, description="Use mock Threads client instead of real API"
    )

    # Memory (Mem0)
    qdrant_url: str = Field(
        default="http://localhost:6333", description="Qdrant vector database URL"
    )
    qdrant_api_key: str | None = Field(
        default=None, description="Qdrant API key"
    )
    database_url: str | None = Field(
        default=None, description="PostgreSQL connection string for Mem0 metadata"
    )

    # Agent Configuration
    agent_name: str = Field(default="AnimaAgent", description="Agent name")
    persona_file: str = Field(
        default="personas/default.json", description="Path to persona definition file"
    )

    # Rate Limiting
    max_daily_posts: int = Field(default=20, description="Maximum posts per day")
    max_daily_replies: int = Field(default=50, description="Maximum replies per day")
    min_interaction_interval_seconds: int = Field(
        default=300, description="Minimum seconds between interactions"
    )

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", description="Logging level"
    )

    # Observation Mode
    observation_mode: bool = Field(
        default=False, description="Run in observation mode (simulate but don't post)"
    )
    simulation_data_dir: str = Field(
        default="data/simulations", description="Directory for simulation data files"
    )

    # External Providers (Apify)
    apify_enabled: bool = Field(default=False, description="Enable Apify provider")
    apify_api_token: str = Field(default="", description="Apify API token")
    apify_actor_id: str = Field(default="", description="Apify actor ID")
    apify_max_age_hours: int = Field(default=24, description="Max age (hours) for Apify posts")
    apify_max_items: int = Field(default=30, description="Max items to fetch from Apify")


def is_reasoning_model(model: str) -> bool:
    """判斷是否為支援 reasoning_effort 的模型。"""
    return "gpt-5" in model or "o1" in model or "o3" in model


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
