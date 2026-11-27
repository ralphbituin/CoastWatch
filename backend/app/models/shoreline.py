"""Shoreline observation and baseline models.

Sign convention used throughout CoastWatch: **positive distance is seaward**.
``shoreline_observations`` is a TimescaleDB hypertable partitioned on
``observed_at``.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.camera import Camera
    from app.models.station import MonitoringStation
    from app.models.tide import TideStation


class ShorelineObservation(Base):
    """A single extracted waterline observation from a camera frame."""

    __tablename__ = "shoreline_observations"
    __table_args__ = (
        Index("ix_shoreline_observations_cam_ts", "camera_id", "observed_at"),
        Index(
            "uq_shoreline_observations_recording_frame",
            "recording_id",
            "frame_index",
            "observed_at",
            unique=True,
        ),
        {"comment": "TimescaleDB hypertable on observed_at"},
    )

    camera_id: Mapped[int] = mapped_column(
        ForeignKey("cameras.id", ondelete="CASCADE"), primary_key=True
    )
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, nullable=False
    )
    monitoring_station_id: Mapped[int] = mapped_column(
        ForeignKey("monitoring_stations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    calibration_id: Mapped[int | None] = mapped_column(
        ForeignKey("camera_calibrations.id", ondelete="SET NULL"), nullable=True
    )

    # List of image-space points [[x, y], ...] for the extracted waterline.
    # Legacy flat field: concatenation of the ordered segments.
    shoreline_geometry: Mapped[list[Any] | None] = mapped_column(JSONB)
    # Ordered polyline segments [[[x, y], ...], ...]. Separate segments are
    # never joined; NULL for historical rows written before this field existed.
    shoreline_segments: Mapped[list[Any] | None] = mapped_column(JSONB)
    # Single-line contract diagnostics (nullable so historical rows remain valid).
    shoreline_axis: Mapped[str | None] = mapped_column(String(12))
    interpolated_ranges: Mapped[list[Any] | None] = mapped_column(JSONB)
    measured_coverage: Mapped[float | None] = mapped_column(Float)
    # Calibrated shoreline position in meters (positive = seaward) when the
    # camera geometry permits calibration; NULL otherwise.
    calibrated_position_m: Mapped[float | None] = mapped_column(Float)

    segmentation_confidence: Mapped[float | None] = mapped_column(Float)
    frame_quality_score: Mapped[float | None] = mapped_column(Float)
    frame_quality_flags: Mapped[list[Any] | None] = mapped_column(JSONB)

    source_frame_path: Mapped[str | None] = mapped_column(String(500))

    # --- Phase 5 video-recording provenance (all nullable, additive) -----
    # Stable recording identity plus frame index make reprocessing idempotent.
    recording_id: Mapped[str | None] = mapped_column(String(64), index=True)
    frame_index: Mapped[int | None] = mapped_column(Integer)
    extraction_status: Mapped[str | None] = mapped_column(String(32))
    extraction_reason: Mapped[str | None] = mapped_column(String(200))
    # Robust representative position for this frame. ``shoreline_position_px``
    # is the raw image coordinate (y for a horizontal shoreline, x for a
    # vertical one). ``signed_position_px`` is the same measurement with the
    # documented sign convention (positive = seaward) so baselines and anomaly
    # scores never have to guess the camera orientation.
    shoreline_position_px: Mapped[float | None] = mapped_column(Float)
    signed_position_px: Mapped[float | None] = mapped_column(Float)
    position_axis: Mapped[str | None] = mapped_column(String(8))
    position_sign: Mapped[str | None] = mapped_column(String(16))
    segment_count: Mapped[int | None] = mapped_column(Integer)
    segment_measurements: Mapped[list[Any] | None] = mapped_column(JSONB)
    class_fractions: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    inference_time_ms: Mapped[float | None] = mapped_column(Float)
    capture_timestamp_source: Mapped[str | None] = mapped_column(String(32))
    capture_timestamp_confidence: Mapped[float | None] = mapped_column(Float)
    # Reference horizon from the live class mask (nullable; NULL means the
    # camera-motion gate is unavailable for this row).
    horizon_y_px: Mapped[float | None] = mapped_column(Float)

    # --- Phase 7 timestamped-recording identity (all nullable, additive) ---
    # Seconds from the source/session start. Archive rows already carry this in
    # CSV output; it was not stored in the database before Phase 7.
    elapsed_seconds: Mapped[float | None] = mapped_column(Float)
    # Live session identity. NULL for archive recordings and historical rows.
    stream_session_id: Mapped[str | None] = mapped_column(String(64), index=True)
    # Live receive-to-persist latency in milliseconds. Never a capture time.
    ingest_latency_ms: Mapped[float | None] = mapped_column(Float)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    camera: Mapped[Camera] = relationship(back_populates="shoreline_observations")
    station: Mapped[MonitoringStation] = relationship(back_populates="shoreline_observations")


class ShorelineBaseline(Base, TimestampMixin):
    """A learned tide-to-shoreline relationship for a camera/station.

    ``expected_shoreline.py`` loads the active baseline to answer "where should
    the waterline be given the tide". Model parameters are stored as JSON so
    new model types can be added without schema changes.
    """

    __tablename__ = "shoreline_baselines"
    __table_args__ = (
        Index(
            "ix_shoreline_baselines_lookup",
            "monitoring_station_id",
            "camera_id",
            "is_active",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    monitoring_station_id: Mapped[int] = mapped_column(
        ForeignKey("monitoring_stations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    camera_id: Mapped[int | None] = mapped_column(
        ForeignKey("cameras.id", ondelete="CASCADE"), nullable=True, index=True
    )
    tide_station_id: Mapped[int | None] = mapped_column(
        ForeignKey("tide_stations.id", ondelete="SET NULL"), nullable=True
    )

    name: Mapped[str] = mapped_column(String(200), nullable=False)
    # "linear" | "polynomial" | "pchip" | ...
    model_type: Mapped[str] = mapped_column(String(32), nullable=False)
    model_params: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    r_squared: Mapped[float | None] = mapped_column(Float)
    residual_mean_m: Mapped[float | None] = mapped_column(Float)
    residual_std_m: Mapped[float | None] = mapped_column(Float)
    # Prediction interval / uncertainty description.
    prediction_interval: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    sample_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # --- Phase 5 prototype metadata (all nullable, additive) -------------
    # "available" | "unavailable"; an unavailable row records *why* instead of
    # pretending a flat/normal baseline exists.
    status: Mapped[str | None] = mapped_column(String(24), nullable=True)
    provisional: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    recording_id: Mapped[str | None] = mapped_column(String(64), index=True)
    position_axis: Mapped[str | None] = mapped_column(String(8))
    position_sign: Mapped[str | None] = mapped_column(String(16))
    position_unit: Mapped[str | None] = mapped_column(String(16))
    # auto | rolling_baseline | tide_conditioned | movement_only
    detector_mode: Mapped[str | None] = mapped_column(String(32), nullable=True)

    valid_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    notes: Mapped[str | None] = mapped_column(Text)

    station: Mapped[MonitoringStation] = relationship(back_populates="baselines")
    camera: Mapped[Camera | None] = relationship()
    tide_station: Mapped[TideStation | None] = relationship()

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<ShorelineBaseline {self.id} model={self.model_type}>"
