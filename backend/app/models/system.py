"""System health log model.

TimescaleDB hypertable partitioned on ``recorded_at``. Used for camera health,
data-quality and pipeline diagnostics.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Float, ForeignKey, Index, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base

__all__ = ["SystemHealthLog"]

# Health levels.
LEVEL_OK = "ok"
LEVEL_WARN = "warn"
LEVEL_ERROR = "error"


class SystemHealthLog(Base):
    __tablename__ = "system_health_logs"
    __table_args__ = (
        Index("ix_system_health_logs_component_ts", "component", "recorded_at"),
        {"comment": "TimescaleDB hypertable on recorded_at"},
    )

    # Composite PK including the time column (TimescaleDB requirement).
    component: Mapped[str] = mapped_column(String(64), primary_key=True)
    metric_name: Mapped[str] = mapped_column(String(64), primary_key=True)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, nullable=False
    )

    monitoring_station_id: Mapped[int | None] = mapped_column(
        ForeignKey("monitoring_stations.id", ondelete="CASCADE"), nullable=True, index=True
    )
    level: Mapped[str] = mapped_column(String(16), nullable=False, default=LEVEL_OK)
    value: Mapped[float | None] = mapped_column(Float)
    message: Mapped[str | None] = mapped_column(Text)
    details: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
