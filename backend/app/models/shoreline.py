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
    shoreline_geometry: Mapped[list[Any] | None] = mapped_column(JSONB)
    # Calibrated shoreline position in meters (positive = seaward) when the
    # camera geometry permits calibration; NULL otherwise.
    calibrated_position_m: Mapped[float | None] = mapped_column(Float)

    segmentation_confidence: Mapped[float | None] = mapped_column(Float)
    frame_quality_score: Mapped[float | None] = mapped_column(Float)
    frame_quality_flags: Mapped[list[Any] | None] = mapped_column(JSONB)

    source_frame_path: Mapped[str | None] = mapped_column(String(500))

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

    valid_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    notes: Mapped[str | None] = mapped_column(Text)

    station: Mapped[MonitoringStation] = relationship(back_populates="baselines")
    camera: Mapped[Camera | None] = relationship()
    tide_station: Mapped[TideStation | None] = relationship()

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<ShorelineBaseline {self.id} model={self.model_type}>"
