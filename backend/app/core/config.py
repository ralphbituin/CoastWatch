"""Application configuration.

All settings are read from environment variables (and an optional ``.env``
file at the repository root). No secrets are hard-coded here — this module
only declares the *names* and sane non-secret defaults.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/core/config.py -> repo root is parents[3]
REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime settings for the CoastWatch backend."""

    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application ---
    project_name: str = "CoastWatch"
    environment: str = Field(default="development")
    debug: bool = Field(default=True)
    api_v1_prefix: str = Field(default="/api/v1")

    # --- Logging ---
    log_level: str = Field(default="INFO")
    # "json" for structured machine-readable logs, "console" for humans.
    log_format: str = Field(default="console")

    # --- CORS (comma separated, e.g. "http://localhost:3000,http://127.0.0.1:3000") ---
    cors_origins: str = Field(default="http://localhost:3000,http://127.0.0.1:3000")

    # --- Database ---
    postgres_user: str = Field(default="coastwatch")
    postgres_password: str = Field(default="coastwatch")
    postgres_db: str = Field(default="coastwatch")
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    # If set, takes precedence over the individual POSTGRES_* variables.
    database_url: str | None = Field(default=None)

    # --- Domain defaults ---
    # IANA time zone used as the default for Philippine monitoring locations.
    default_timezone: str = Field(default="Asia/Manila")
    # Directory that holds trained model artifacts (Phase 4+). Never commit weights.
    model_path: str = Field(default=str(REPO_ROOT / "ml" / "models"))
    # Root directory for operator-provided datasets (CSV tide tables, videos).
    data_dir: str = Field(default=str(REPO_ROOT / "data"))

    @property
    def sync_database_url(self) -> str:
        """SQLAlchemy URL for the (synchronous) PostgreSQL + PostGIS database."""
        if self.database_url:
            return self.database_url
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def cors_origins_list(self) -> list[str]:
        """CORS origins as a list."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance."""
    return Settings()


settings = get_settings()
