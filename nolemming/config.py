"""Global configuration for NoLemming."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class NoLemmingConfig(BaseSettings):
    """Configuration loaded from environment variables or .env file.

    All LLM and encoder settings are plug-and-play:
    set FARSWARM_LLM_MODEL to any model, FARSWARM_LLM_BASE_URL to any
    OpenAI-compatible endpoint (Ollama, vLLM, Together, etc.).
    """

    model_config = {"env_prefix": "FARSWARM_"}

    # LLM settings — works with any OpenAI SDK-compatible provider
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str | None = None
    llm_base_url: str | None = None

    # Brain encoder
    encoder_name: str = "mock"
    tribe_v2_cache: Path = Path("./cache/tribe_v2")

    # Simulation defaults
    default_n_agents: int = 1000
    default_n_rounds: int = 168
    default_minutes_per_round: int = 60
    default_n_archetypes: int = 8
    default_pca_dims: int = 64

    # Embedding model for content similarity
    embedding_model: str = "all-MiniLM-L6-v2"

    # Output
    output_dir: Path = Path("./output")
    benchmark_data_dir: Path = Path("./benchmarks/data")

    # Concurrency
    max_concurrent_llm_calls: int = 30
