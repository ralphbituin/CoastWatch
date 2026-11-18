"""Environmental observations (wave, wind, rain, visibility, ...).

TimescaleDB hypertable partitioned on ``observed_at``. Optional in the pipeline
but useful context for the baseline model (Phase 5).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Float, ForeignKey, Index, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base

__all__ = ["EnvironmentalObservation"]


class EnvironmentalObservation(Base):
    __tablename__ = "environmental_observations"
    __table_args__ = (
        Index("ix_environmental_obs_station_ts", "monitoring_station_id", "observed_at"),
        {"comment": "TimescaleDB hypertable on observed_at"},
    )

    monitoring_station_id: Mapped[int] = mapped_column(
        ForeignKey("monitoring_stations.id", ondelete="CASCADE"), primary_key=True
    )
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, nullable=False
    )

    wave_height_m: Mapped[float | None] = mapped_column(Float)
    wind_speed_m_s: Mapped[float | None] = mapped_column(Float)
    wind_direction_deg: Mapped[float | None] = mapped_column(Float)
    rainfall_mm: Mapped[float | None] = mapped_column(Float)
    visibility_m: Mapped[float | None] = mapped_column(Float)

    source: Mapped[str | None] = mapped_column(String(64))
    extra: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
