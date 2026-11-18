"""Country reference table.

Supports the multi-country design goal (Philippines first, Southeast Asia
later) without assuming shared time zones, datums or tide providers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.station import MonitoringStation
    from app.models.tide import TideStation


class Country(Base, TimestampMixin):
    __tablename__ = "countries"

    id: Mapped[int] = mapped_column(primary_key=True)
    # ISO 3166-1 alpha-2 (e.g. "PH").
    iso_code: Mapped[str] = mapped_column(String(2), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    # Default IANA time zone for the country (per-location overrides allowed).
    default_timezone: Mapped[str] = mapped_column(String(64), nullable=False, default="UTC")

    monitoring_stations: Mapped[list[MonitoringStation]] = relationship(back_populates="country")
    tide_stations: Mapped[list[TideStation]] = relationship(back_populates="country")

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Country {self.iso_code}>"
