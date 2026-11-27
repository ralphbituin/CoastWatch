"""phase 6 rev 2: tide-independent rolling-baseline metadata

Revision ID: c7d9a1b3e5f2
Revises: b2c5e8f1d4a6
Create Date: 2026-11-25 00:00:00.000000

Additive and nullable only. Historical rows are not rewritten.

``anomaly_events.classification`` remains a plain ``String(48)`` (not a
PostgreSQL enum, not CHECK-constrained), so no ``ALTER TYPE`` is required and
the downgrade can remove the new index/columns cleanly. If a future deployment
converts this column to a PostgreSQL enum, ``ALTER TYPE ... ADD VALUE`` cannot
be rolled back and must be documented separately.

``anomaly_events`` is a TimescaleDB hypertable on ``detected_at``. TimescaleDB
requires the partition column in a unique index, so the logical idempotency key
``(recording_id, detector_mode, onset_frame_index)`` includes ``detected_at``.
The detection timestamp is deterministic for a given event, so re-processing
still cannot duplicate an event.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c7d9a1b3e5f2"
down_revision: str | None = "b2c5e8f1d4a6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "shoreline_observations",
        sa.Column("horizon_y_px", sa.Float(), nullable=True),
    )
    op.add_column(
        "shoreline_baselines",
        sa.Column("detector_mode", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "anomaly_events",
        sa.Column("detector_mode", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "anomaly_events",
        sa.Column("onset_frame_index", sa.Integer(), nullable=True),
    )
    op.create_index(
        "ix_anomaly_events_detector_mode",
        "anomaly_events",
        ["detector_mode"],
        unique=False,
    )
    op.create_index(
        "uq_anomaly_events_recording_mode_onset",
        "anomaly_events",
        ["recording_id", "detector_mode", "onset_frame_index", "detected_at"],
        unique=True,
        postgresql_where=sa.text(
            "recording_id IS NOT NULL AND detector_mode IS NOT NULL "
            "AND onset_frame_index IS NOT NULL"
        ),
    )


def downgrade() -> None:
    op.drop_index("uq_anomaly_events_recording_mode_onset", table_name="anomaly_events")
    op.drop_index("ix_anomaly_events_detector_mode", table_name="anomaly_events")
    op.drop_column("anomaly_events", "onset_frame_index")
    op.drop_column("anomaly_events", "detector_mode")
    op.drop_column("shoreline_baselines", "detector_mode")
    op.drop_column("shoreline_observations", "horizon_y_px")
