"""Camera and camera-calibration models.

Cameras are never assumed to share resolution, frame rate, orientation or
calibration. Each camera keeps its own calibration profile history.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from geoalchemy2 import Geography
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.anomaly import AnomalyEvent
    from app.models.shoreline import ShorelineObservation
    from app.models.station import MonitoringStation


class Camera(Base, TimestampMixin):
    """A configured camera feeding a monitoring station."""

    __tablename__ = "cameras"

    id: Mapped[int] = mapped_column(primary_key=True)
    monitoring_station_id: Mapped[int] = mapped_column(
        ForeignKey("monitoring_stations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    code: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)

    # "rtsp" | "usb" | "upload" | "video"
    camera_type: Mapped[str] = mapped_column(String(16), nullable=False)

    # RTSP transport (nullable — never log the full URL with credentials).
    rtsp_url: Mapped[str | None] = mapped_column(String(500))
    # USB device index (OpenCV VideoCapture index).
    device_index: Mapped[int | None] = mapped_column(Integer)
    # Virtual video camera: path relative to DATA_DIR/videos/ (Phase 4 dev tooling).
    video_path: Mapped[str | None] = mapped_column(String(500))

    resolution_width: Mapped[int | None] = mapped_column(Integer)
    resolution_height: Mapped[int | None] = mapped_column(Integer)
    fps: Mapped[float | None] = mapped_column(Float)

    latitude: Mapped[float | None] = mapped_column(Float)
    longitude: Mapped[float | None] = mapped_column(Float)
    location: Mapped[str | None] = mapped_column(
        Geography(geometry_type="POINT", srid=4326), nullable=True
    )

    # Free-text / numeric orientation description (bearing in degrees, etc.).
    orientation_deg: Mapped[float | None] = mapped_column(Float)
    # Region of interest / monitoring region as a GeoJSON-like structure.
    monitoring_region: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    active_calibration_id: Mapped[int | None] = mapped_column(
        ForeignKey(
            "camera_calibrations.id",
            ondelete="SET NULL",
            use_alter=True,
            name="fk_cameras_active_calibration_id_camera_calibrations",
        ),
        nullable=True,
        index=True,
    )
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    station: Mapped[MonitoringStation] = relationship(back_populates="cameras")
    calibrations: Mapped[list[CameraCalibration]] = relationship(
        back_populates="camera",
        foreign_keys="CameraCalibration.camera_id",
    )
    shoreline_observations: Mapped[list[ShorelineObservation]] = relationship(
        back_populates="camera"
    )
    anomaly_events: Mapped[list[AnomalyEvent]] = relationship(back_populates="camera")

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Camera {self.code} type={self.camera_type}>"


class CameraCalibration(Base, TimestampMixin):
    """A versioned homography / scale calibration for one camera."""

    __tablename__ = "camera_calibrations"
    __table_args__ = (
        Index("ix_camera_calibrations_camera_version", "camera_id", "version", unique=True),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    camera_id: Mapped[int] = mapped_column(
        ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False, index=True
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    # Operator-supplied reference points: pixel (x,y) <-> real-world (x,y, meters).
    reference_points: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    # Row-major 3x3 homography matrix.
    homography_matrix: Mapped[list[Any] | None] = mapped_column(JSONB)
    meters_per_pixel: Mapped[float | None] = mapped_column(Float)

    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    valid_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    notes: Mapped[str | None] = mapped_column(Text)

    camera: Mapped[Camera] = relationship(back_populates="calibrations", foreign_keys=[camera_id])

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<CameraCalibration cam={self.camera_id} v{self.version}>"
