"""Verify the ORM metadata defines every required entity.

This test needs no database connection: it inspects ``Base.metadata``.
"""

from __future__ import annotations

from app.database.base import Base, import_models

import_models()

EXPECTED_TABLES = {
    "countries",
    "monitoring_stations",
    "tide_stations",
    "tide_datasets",
    "tide_predictions",
    "dataset_imports",
    "cameras",
    "camera_calibrations",
    "shoreline_observations",
    "shoreline_baselines",
    "anomaly_events",
    "environmental_observations",
    "system_health_logs",
    "video_recordings",
}


def test_all_required_tables_registered() -> None:
    assert EXPECTED_TABLES.issubset(set(Base.metadata.tables))


def test_hypertable_tables_have_time_column_in_primary_key() -> None:
    """TimescaleDB requires the partition (time) column in the primary key."""
    hypertables = {
        "tide_predictions": "timestamp",
        "shoreline_observations": "observed_at",
        "system_health_logs": "recorded_at",
        "anomaly_events": "detected_at",
        "environmental_observations": "observed_at",
    }
    for table_name, time_column in hypertables.items():
        table = Base.metadata.tables[table_name]
        pk_columns = {c.name for c in table.primary_key.columns}
        assert time_column in pk_columns, f"{table_name} PK must include {time_column}"


def test_geography_columns_present() -> None:
    for table_name in ("monitoring_stations", "cameras", "tide_stations"):
        table = Base.metadata.tables[table_name]
        assert "location" in table.columns


def test_orm_mappers_configure() -> None:
    """Configure all mappers so relationship targets actually resolve.

    This catches unresolved string annotations in ``Mapped[...]`` relationships
    that ``Base.metadata`` inspection alone would miss.
    """
    from sqlalchemy.orm import configure_mappers

    configure_mappers()
