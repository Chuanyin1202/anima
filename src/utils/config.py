"""Configuration management using Pydantic Settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # OpenAI
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o-mini", description="Default OpenAI model")
    openai_model_advanced: str = Field(
        default="gpt-4o", description="Advanced model for complex tasks"
    )

    # Threads API
    threads_app_id: str = Field(default="", description="Meta Threads App ID")
    threads_app_secret: str = Field(default="", description="Meta Threads App Secret")
    threads_access_token: str = Field(default="", description="Threads access token")
    threads_user_id: str = Field(default="", description="Threads user ID")

    # Mock Mode (for testing without real API)
    use_mock_threads: bool = Field(
        default=False, description="Use mock Threads client instead of real API"
    )

    # Memory (Mem0)
    qdrant_url: str = Field(
        default="http://localhost:6333", description="Qdrant vector database URL"
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


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
