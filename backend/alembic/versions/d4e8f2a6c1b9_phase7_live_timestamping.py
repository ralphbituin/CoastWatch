"""phase 7: timestamped live recording, YouTube Live cameras, stream sessions

Revision ID: d4e8f2a6c1b9
Revises: c7d9a1b3e5f2
Create Date: 2026-11-25 12:00:00.000000

Additive and nullable only. Historical observations and events are not
rewritten. ``classification`` remains a plain ``String(48)``; no PostgreSQL enum
is introduced, so no non-reversible ``ALTER TYPE`` is needed.

New ``stream_sessions`` rows are never TimescaleDB hypertables (bounded session
metadata, one row per capture run).
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "d4e8f2a6c1b9"
down_revision: str | None = "c7d9a1b3e5f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # --- shoreline observations: per-timestamp live provenance ------------
    op.add_column(
        "shoreline_observations",
        sa.Column("elapsed_seconds", sa.Float(), nullable=True),
    )
    op.add_column(
        "shoreline_observations",
        sa.Column("stream_session_id", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "shoreline_observations",
        sa.Column("ingest_latency_ms", sa.Float(), nullable=True),
    )
    op.create_index(
        "ix_shoreline_observations_stream_session_id",
        "shoreline_observations",
        ["stream_session_id"],
        unique=False,
    )

    # --- cameras: YouTube Live source (semantics of rtsp_url unchanged) ---
    op.add_column("cameras", sa.Column("youtube_url", sa.String(length=500), nullable=True))
    op.add_column("cameras", sa.Column("stream_quality", sa.String(length=32), nullable=True))
    op.add_column("cameras", sa.Column("stream_format", sa.String(length=32), nullable=True))

    # --- live stream sessions ---------------------------------------------
    op.create_table(
        "stream_sessions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("recording_id", sa.String(length=64), nullable=False),
        sa.Column("camera_id", sa.Integer(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("source_url_redacted", sa.String(length=500), nullable=True),
        sa.Column("sample_interval_seconds", sa.Float(), nullable=True),
        sa.Column("last_frame_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("frames_captured", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("frames_persisted", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("frames_rejected", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("reconnect_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("last_rolling_classification", sa.String(length=48), nullable=True),
        sa.Column("last_event_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("detector_config", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
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
            name="fk_stream_sessions_camera_id_cameras",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_stream_sessions"),
        sa.UniqueConstraint("session_id", name="uq_stream_sessions_session_id"),
    )
    op.create_index(
        "ix_stream_sessions_camera_started",
        "stream_sessions",
        ["camera_id", "started_at"],
        unique=False,
    )
    op.create_index(
        "ix_stream_sessions_recording_id",
        "stream_sessions",
        ["recording_id"],
        unique=False,
    )
    op.create_index(
        "ix_stream_sessions_camera_id",
        "stream_sessions",
        ["camera_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_stream_sessions_camera_id", table_name="stream_sessions")
    op.drop_index("ix_stream_sessions_recording_id", table_name="stream_sessions")
    op.drop_index("ix_stream_sessions_camera_started", table_name="stream_sessions")
    op.drop_table("stream_sessions")

    op.drop_column("cameras", "stream_format")
    op.drop_column("cameras", "stream_quality")
    op.drop_column("cameras", "youtube_url")

    op.drop_index(
        "ix_shoreline_observations_stream_session_id",
        table_name="shoreline_observations",
    )
    op.drop_column("shoreline_observations", "ingest_latency_ms")
    op.drop_column("shoreline_observations", "stream_session_id")
    op.drop_column("shoreline_observations", "elapsed_seconds")
