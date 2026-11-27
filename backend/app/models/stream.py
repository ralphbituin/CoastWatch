"""Live stream session identity (Phase 7, additive).

A live session is created **once** when a capture worker starts and keeps the
same ``session_id`` / ``recording_id`` for every sampled observation. It is never
regenerated per frame, so ``(recording_id, frame_index)`` remains a stable
idempotency key for live rows and rolling events.

``source_url_redacted`` never contains credentials, cookies or query tokens.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.camera import Camera

# Stream lifecycle states exposed by the API and persisted for recovery.
STREAM_STOPPED = "stopped"
STREAM_STARTING = "starting"
STREAM_RUNNING = "running"
STREAM_RECONNECTING = "reconnecting"
STREAM_FAILED = "failed"

STREAM_STATES = {
    STREAM_STOPPED,
    STREAM_STARTING,
    STREAM_RUNNING,
    STREAM_RECONNECTING,
    STREAM_FAILED,
}


class StreamSession(Base, TimestampMixin):
    """One bounded live-capture session for a camera."""

    __tablename__ = "stream_sessions"
    __table_args__ = (Index("ix_stream_sessions_camera_started", "camera_id", "started_at"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    # Stable identities generated at start, not per frame.
    session_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    recording_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    camera_id: Mapped[int] = mapped_column(
        ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False, index=True
    )

    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(24), nullable=False, default=STREAM_STARTING)

    # Redacted public/authorized source URL. Never a credentialed URL.
    source_url_redacted: Mapped[str | None] = mapped_column(String(500))
    sample_interval_seconds: Mapped[float | None] = mapped_column(Float)

    last_frame_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    frames_captured: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    frames_persisted: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    frames_rejected: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    reconnect_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    last_error: Mapped[str | None] = mapped_column(Text)
    last_rolling_classification: Mapped[str | None] = mapped_column(String(48))
    last_event_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    detector_config: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    camera: Mapped[Camera] = relationship(back_populates="stream_sessions")

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<StreamSession {self.session_id} status={self.status}>"
