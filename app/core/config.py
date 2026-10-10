"""Application settings, loaded from environment variables (.env).

One Settings object. Holds the backend selection for each model role
(generator/embedder/reranker/transcriber), database and cache URLs, and
the latency/score-floor tunables from the query pipeline. Changing a
model role from cloud to local happens here, in config, never in code
that calls the model layer.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Backend(StrEnum):
    CLOUD = "cloud"
    LOCAL = "local"


class RoleSettings(BaseSettings):
    backend: Backend = Backend.CLOUD
    model: str = ""
    api_key: str = ""
    base_url: str = ""

    def configured(self) -> bool:
        if self.backend is Backend.LOCAL:
            return bool(self.model)
        return bool(self.model and self.api_key)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    database_url: str

    generator: RoleSettings = Field(default_factory=RoleSettings)
    embedder: RoleSettings = Field(default_factory=RoleSettings)
    reranker: RoleSettings = Field(default_factory=RoleSettings)
    transcriber: RoleSettings = Field(default_factory=RoleSettings)

    local_model_base_url: str = "http://localhost:11434/v1"

    ingestion_source_dir: Path = Path("./Documents")
    upload_max_file_mb: int = 50

    # Placeholder until phase 2 calibrates it against the eval set. The
    # scale is reranker-specific, so this number is meaningless without
    # knowing which reranker produced the scores it gates.
    retrieval_score_floor: float = 0.5
    query_latency_budget_ms: int = 3000

    cache_url: str = ""
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def settings() -> Settings:
    return Settings()
