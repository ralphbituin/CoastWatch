"""Video-recording provenance for offline/Phase 5 prototype processing.

A recording is identified by a stable ``recording_id`` derived from the file
identity (path + size + mtime), never from the misleading filename alone.
``camera_id`` is nullable because an archive can be analysed before an operator
registers it as a camera. Nothing here fabricates capture timestamps: the
container may have no creation time, in which case ``timestamp_source`` records
``unavailable`` or ``derived`` and the clock anchors are kept verbatim.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import BigInteger, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.camera import Camera
    from app.models.station import MonitoringStation


class VideoRecording(Base, TimestampMixin):
    """Metadata and processing status for one video archive."""

    __tablename__ = "video_recordings"
    __table_args__ = (Index("ix_video_recordings_camera_created", "camera_id", "created_at"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    # Stable identity: independent of the (possibly wrong) filename.
    recording_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    camera_id: Mapped[int | None] = mapped_column(
        ForeignKey("cameras.id", ondelete="SET NULL"), nullable=True, index=True
    )
    monitoring_station_id: Mapped[int | None] = mapped_column(
        ForeignKey("monitoring_stations.id", ondelete="SET NULL"), nullable=True, index=True
    )

    file_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    file_size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    file_sha256: Mapped[str | None] = mapped_column(String(64))

    duration_seconds: Mapped[float | None] = mapped_column(Float)
    fps: Mapped[float | None] = mapped_column(Float)
    frame_count: Mapped[int | None] = mapped_column(Integer)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    video_codec: Mapped[str | None] = mapped_column(String(32))
    audio_codec: Mapped[str | None] = mapped_column(String(32))

    timezone: Mapped[str | None] = mapped_column(String(64))
    timestamp_source: Mapped[str | None] = mapped_column(String(32))
    timestamp_confidence: Mapped[float | None] = mapped_column(Float)
    # Verified visible-clock anchors: [{elapsed_seconds, timestamp, source,
    # confidence, verification}]. No OCR is implied when verification is manual.
    clock_anchors: Mapped[list[Any] | None] = mapped_column(JSONB)

    processing_status: Mapped[str | None] = mapped_column(String(32), default="registered")
    processed_frames: Mapped[int | None] = mapped_column(Integer, default=0)
    sampled_frames: Mapped[int | None] = mapped_column(Integer, default=0)
    # Container tags / ffprobe output / processing diagnostics. Kept separate
    # from SQLAlchemy's reserved ``metadata`` attribute.
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    notes: Mapped[str | None] = mapped_column(Text)

    camera: Mapped[Camera | None] = relationship()
    station: Mapped[MonitoringStation | None] = relationship()

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<VideoRecording {self.recording_id} status={self.processing_status}>"
