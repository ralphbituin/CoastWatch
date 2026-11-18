"""Anomaly event model.

``anomaly_events`` is a TimescaleDB hypertable partitioned on ``detected_at``.
The composite primary key ``(id, detected_at)`` includes the partition column,
as required by TimescaleDB.

The system states are deliberately conservative. ``OFFICIAL_TSUNAMI_ADVISORY``
records an independently verified advisory; the anomaly engine itself must
never claim to confirm a tsunami.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Identity,
    Index,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base

if TYPE_CHECKING:
    from app.models.camera import Camera
    from app.models.station import MonitoringStation

# System states.
STATE_NORMAL = "NORMAL"
STATE_ANOMALY = "ANOMALY_DETECTED"
STATE_UNRELIABLE = "DATA_UNRELIABLE"
STATE_OFFICIAL = "OFFICIAL_TSUNAMI_ADVISORY"


class AnomalyEvent(Base):
    """A persisted anomaly / reliability event with an explanation."""

    __tablename__ = "anomaly_events"
    __table_args__ = (
        Index("ix_anomaly_events_station_ts", "monitoring_station_id", "detected_at"),
        {"comment": "TimescaleDB hypertable on detected_at"},
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True, autoincrement=True)
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, nullable=False
    )

    monitoring_station_id: Mapped[int] = mapped_column(
        ForeignKey("monitoring_stations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    camera_id: Mapped[int | None] = mapped_column(
        ForeignKey("cameras.id", ondelete="SET NULL"), nullable=True, index=True
    )

    state: Mapped[str] = mapped_column(String(32), nullable=False, default=STATE_NORMAL)
    anomaly_score: Mapped[float | None] = mapped_column(Float)

    residual_m: Mapped[float | None] = mapped_column(Float)
    retreat_velocity_m_s: Mapped[float | None] = mapped_column(Float)
    retreat_acceleration_m_s2: Mapped[float | None] = mapped_column(Float)
    persistence: Mapped[float | None] = mapped_column(Float)
    baseline_uncertainty_m: Mapped[float | None] = mapped_column(Float)

    tide_phase: Mapped[str | None] = mapped_column(String(32))
    tide_height_m: Mapped[float | None] = mapped_column(Float)
    expected_shoreline_m: Mapped[float | None] = mapped_column(Float)
    actual_shoreline_m: Mapped[float | None] = mapped_column(Float)

    explanation: Mapped[str | None] = mapped_column(Text)
    contributing_factors: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    acknowledged: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    station: Mapped[MonitoringStation] = relationship(back_populates="anomaly_events")
    camera: Mapped[Camera | None] = relationship(back_populates="anomaly_events")

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<AnomalyEvent {self.id} state={self.state}>"
