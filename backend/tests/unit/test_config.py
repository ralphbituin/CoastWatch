"""Tests for application settings."""

from __future__ import annotations

from app.core.config import Settings


def test_default_timezone_is_manila() -> None:
    settings = Settings()
    assert settings.default_timezone == "Asia/Manila"


def test_database_url_from_parts() -> None:
    settings = Settings(
        postgres_user="u",
        postgres_password="p",
        postgres_host="db",
        postgres_port=5432,
        postgres_db="coastwatch",
        database_url=None,
    )
    assert settings.sync_database_url == "postgresql+psycopg2://u:p@db:5432/coastwatch"


def test_database_url_override_wins() -> None:
    override = "postgresql+psycopg2://x:y@z:5432/w"
    settings = Settings(database_url=override)
    assert settings.sync_database_url == override


def test_cors_origins_parsed_into_list() -> None:
    settings = Settings(cors_origins="http://a, http://b ,")
    assert settings.cors_origins_list == ["http://a", "http://b"]
