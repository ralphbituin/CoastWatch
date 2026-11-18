"""Monitoring stations: a physical coastal site being watched.

A monitoring station groups cameras and is linked to one or more tide
stations. Its geographic position is stored both as plain lat/lon floats and
as a PostGIS geography point for spatial queries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from geoalchemy2 import Geography
from sqlalchemy import Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.anomaly import AnomalyEvent
    from app.models.camera import Camera
    from app.models.country import Country
    from app.models.shoreline import ShorelineBaseline, ShorelineObservation
    from app.models.tide import TideStation


class MonitoringStation(Base, TimestampMixin):
    __tablename__ = "monitoring_stations"

    id: Mapped[int] = mapped_column(primary_key=True)
    country_id: Mapped[int] = mapped_column(
        ForeignKey("countries.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    code: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    region: Mapped[str | None] = mapped_column(String(120))
    description: Mapped[str | None] = mapped_column(Text)

    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    # PostGIS geography point (WGS84) for spatial queries / distances.
    location: Mapped[str | None] = mapped_column(
        Geography(geometry_type="POINT", srid=4326), nullable=True
    )

    timezone: Mapped[str] = mapped_column(String(64), nullable=False, default="Asia/Manila")
    # Vertical datum reference for any elevation/datum-sensitive data here.
    vertical_datum: Mapped[str | None] = mapped_column(String(64))

    country: Mapped[Country] = relationship(back_populates="monitoring_stations")
    cameras: Mapped[list[Camera]] = relationship(back_populates="station")
    tide_stations: Mapped[list[TideStation]] = relationship(back_populates="monitoring_station")
    shoreline_observations: Mapped[list[ShorelineObservation]] = relationship(
        back_populates="station"
    )
    baselines: Mapped[list[ShorelineBaseline]] = relationship(back_populates="station")
    anomaly_events: Mapped[list[AnomalyEvent]] = relationship(back_populates="station")

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<MonitoringStation {self.code}>"
