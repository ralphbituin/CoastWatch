"""phase 5/6 video recording + tide-aware anomaly prototype

Revision ID: b2c5e8f1d4a6
Revises: a1f4c2d9e7b3
Create Date: 2026-11-24 12:00:00.000000

All additions are nullable and additive. Existing observations, baselines and
events are preserved untouched. No table is rewritten and no column is dropped.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "b2c5e8f1d4a6"
down_revision: str | None = "a1f4c2d9e7b3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "video_recordings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("recording_id", sa.String(length=64), nullable=False),
        sa.Column("camera_id", sa.Integer(), nullable=True),
        sa.Column("monitoring_station_id", sa.Integer(), nullable=True),
        sa.Column("file_path", sa.String(length=1000), nullable=False),
        sa.Column("file_size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("file_sha256", sa.String(length=64), nullable=True),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column("fps", sa.Float(), nullable=True),
        sa.Column("frame_count", sa.Integer(), nullable=True),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("video_codec", sa.String(length=32), nullable=True),
        sa.Column("audio_codec", sa.String(length=32), nullable=True),
        sa.Column("timezone", sa.String(length=64), nullable=True),
        sa.Column("timestamp_source", sa.String(length=32), nullable=True),
        sa.Column("timestamp_confidence", sa.Float(), nullable=True),
        sa.Column("clock_anchors", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("processing_status", sa.String(length=32), nullable=True),
        sa.Column("processed_frames", sa.Integer(), nullable=True),
        sa.Column("sampled_frames", sa.Integer(), nullable=True),
        sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["camera_id"],
            ["cameras.id"],
            name="fk_video_recordings_camera_id_cameras",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["monitoring_station_id"],
            ["monitoring_stations.id"],
            name="fk_video_recordings_monitoring_station_id_monitoring_stations",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_video_recordings"),
        sa.UniqueConstraint("recording_id", name="uq_video_recordings_recording_id"),
    )
    op.create_index(
        "ix_video_recordings_camera_created",
        "video_recordings",
        ["camera_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_video_recordings_camera_id", "video_recordings", ["camera_id"], unique=False
    )
    op.create_index(
        "ix_video_recordings_monitoring_station_id",
        "video_recordings",
        ["monitoring_station_id"],
        unique=False,
    )

    # --- shoreline_observations -------------------------------------------
    for column in (
        sa.Column("recording_id", sa.String(length=64), nullable=True),
        sa.Column("frame_index", sa.Integer(), nullable=True),
        sa.Column("extraction_status", sa.String(length=32), nullable=True),
        sa.Column("extraction_reason", sa.String(length=200), nullable=True),
        sa.Column("shoreline_position_px", sa.Float(), nullable=True),
        sa.Column("signed_position_px", sa.Float(), nullable=True),
        sa.Column("position_axis", sa.String(length=8), nullable=True),
        sa.Column("position_sign", sa.String(length=16), nullable=True),
        sa.Column("segment_count", sa.Integer(), nullable=True),
        sa.Column("segment_measurements", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("class_fractions", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("inference_time_ms", sa.Float(), nullable=True),
        sa.Column("capture_timestamp_source", sa.String(length=32), nullable=True),
        sa.Column("capture_timestamp_confidence", sa.Float(), nullable=True),
    ):
        op.add_column("shoreline_observations", column)
    op.create_index(
        "ix_shoreline_observations_recording_id",
        "shoreline_observations",
        ["recording_id"],
        unique=False,
    )
    op.create_index(
        "uq_shoreline_observations_recording_frame",
        "shoreline_observations",
        ["recording_id", "frame_index", "observed_at"],
        unique=True,
    )

    # --- shoreline_baselines ----------------------------------------------
    for column in (
        sa.Column("status", sa.String(length=24), nullable=True),
        sa.Column("provisional", sa.Boolean(), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("recording_id", sa.String(length=64), nullable=True),
        sa.Column("position_axis", sa.String(length=8), nullable=True),
        sa.Column("position_sign", sa.String(length=16), nullable=True),
        sa.Column("position_unit", sa.String(length=16), nullable=True),
    ):
        op.add_column("shoreline_baselines", column)
    op.create_index(
        "ix_shoreline_baselines_recording_id",
        "shoreline_baselines",
        ["recording_id"],
        unique=False,
    )

    # --- anomaly_events ---------------------------------------------------
    for column in (
        sa.Column("recording_id", sa.String(length=64), nullable=True),
        sa.Column("frame_index", sa.Integer(), nullable=True),
        sa.Column("classification", sa.String(length=48), nullable=True),
        sa.Column("baseline_status", sa.String(length=24), nullable=True),
        sa.Column("tide_adjusted", sa.Boolean(), nullable=True),
        sa.Column("tide_data_available", sa.Boolean(), nullable=True),
        sa.Column("tide_source", sa.String(length=64), nullable=True),
        sa.Column("robust_z", sa.Float(), nullable=True),
        sa.Column("observed_position_px", sa.Float(), nullable=True),
        sa.Column("expected_position_px", sa.Float(), nullable=True),
        sa.Column("residual_px", sa.Float(), nullable=True),
        sa.Column("movement_rate_px_s", sa.Float(), nullable=True),
        sa.Column("persistence_count", sa.Integer(), nullable=True),
        sa.Column("quality_status", sa.String(length=32), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
    ):
        op.add_column("anomaly_events", column)
    op.create_index(
        "ix_anomaly_events_recording_id", "anomaly_events", ["recording_id"], unique=False
    )
    op.create_index(
        "ix_anomaly_events_classification",
        "anomaly_events",
        ["classification"],
        unique=False,
    )
    op.create_index(
        "uq_anomaly_events_recording_frame",
        "anomaly_events",
        ["recording_id", "frame_index", "detected_at"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("uq_anomaly_events_recording_frame", table_name="anomaly_events")
    op.drop_index("ix_anomaly_events_classification", table_name="anomaly_events")
    op.drop_index("ix_anomaly_events_recording_id", table_name="anomaly_events")
    for name in (
        "reason",
        "quality_status",
        "persistence_count",
        "movement_rate_px_s",
        "residual_px",
        "expected_position_px",
        "observed_position_px",
        "robust_z",
        "tide_source",
        "tide_data_available",
        "tide_adjusted",
        "baseline_status",
        "classification",
        "frame_index",
        "recording_id",
    ):
        op.drop_column("anomaly_events", name)

    op.drop_index("ix_shoreline_baselines_recording_id", table_name="shoreline_baselines")
    for name in (
        "position_unit",
        "position_sign",
        "position_axis",
        "recording_id",
        "reason",
        "provisional",
        "status",
    ):
        op.drop_column("shoreline_baselines", name)

    op.drop_index("uq_shoreline_observations_recording_frame", table_name="shoreline_observations")
    op.drop_index("ix_shoreline_observations_recording_id", table_name="shoreline_observations")
    for name in (
        "capture_timestamp_confidence",
        "capture_timestamp_source",
        "inference_time_ms",
        "class_fractions",
        "segment_measurements",
        "segment_count",
        "position_sign",
        "position_axis",
        "signed_position_px",
        "shoreline_position_px",
        "extraction_reason",
        "extraction_status",
        "frame_index",
        "recording_id",
    ):
        op.drop_column("shoreline_observations", name)

    op.drop_index("ix_video_recordings_monitoring_station_id", table_name="video_recordings")
    op.drop_index("ix_video_recordings_camera_id", table_name="video_recordings")
    op.drop_index("ix_video_recordings_camera_created", table_name="video_recordings")
    op.drop_table("video_recordings")
