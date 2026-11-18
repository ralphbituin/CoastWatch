"""Alembic migration environment.

The database URL and target metadata come from the application, so migrations
and models can never drift apart. Autogenerate compares against
``Base.metadata`` populated by :func:`app.database.base.import_models`.
"""

from __future__ import annotations

from alembic import context
from sqlalchemy import engine_from_config, pool

from app.core.config import settings
from app.core.logging import configure_logging
from app.database.base import Base, import_models

# Ensure all models are registered on the metadata.
import_models()

config = context.config
config.set_main_option("sqlalchemy.url", settings.sync_database_url)

# Use the application's logging configuration rather than a separate
# alembic.ini logging section.
configure_logging()

target_metadata = Base.metadata

# Objects that legitimately exist in the database but are not part of the
# application model: PostGIS/extension internals and TimescaleDB-managed
# indexes on hypertable partition columns.
_IGNORED_TABLES = {"spatial_ref_sys"}
_IGNORED_INDEXES = {
    "tide_predictions_timestamp_idx",
    "shoreline_observations_observed_at_idx",
    "anomaly_events_detected_at_idx",
    "system_health_logs_recorded_at_idx",
    "environmental_observations_observed_at_idx",
}


def include_object(object_, name, type_, reflected, compare_to):  # noqa: ANN001
    """Filter out extension-managed objects so autogenerate/check stay clean."""
    return not (
        (type_ == "table" and name in _IGNORED_TABLES)
        or (type_ == "index" and name in _IGNORED_INDEXES)
    )


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL without a DB connection)."""
    context.configure(
        url=settings.sync_database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        include_object=include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode (with a DB connection)."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            include_object=include_object,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
