"""Tide-domain models.

IMPORTANT design rules (see ``docs/DATA_QUALITY.md``):

* Predicted tide heights are kept **separate** from measured sea levels.
* Each tide station keeps its own vertical datum and unit; datums are never
  blended across stations.
* Imported datasets and their import history are insert-only — a dataset is
  never overwritten in place.
* ``tide_predictions`` is a TimescaleDB hypertable partitioned on ``timestamp``.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from geoalchemy2 import Geography
from sqlalchemy import (
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
    from app.models.country import Country
    from app.models.station import MonitoringStation


class TideStation(Base, TimestampMixin):
    """A tide-prediction reference point (e.g. a NAMRIA station)."""

    __tablename__ = "tide_stations"

    id: Mapped[int] = mapped_column(primary_key=True)
    country_id: Mapped[int] = mapped_column(
        ForeignKey("countries.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    # Optional link: a monitoring station may observe the sea level predicted
    # by a specific tide station. Left nullable on purpose — never assume a
    # 1:1 mapping, and never silently substitute a different station.
    monitoring_station_id: Mapped[int | None] = mapped_column(
        ForeignKey("monitoring_stations.id", ondelete="SET NULL"), nullable=True, index=True
    )
    code: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    region: Mapped[str | None] = mapped_column(String(120))

    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    location: Mapped[str | None] = mapped_column(
        Geography(geometry_type="POINT", srid=4326), nullable=True
    )

    timezone: Mapped[str] = mapped_column(String(64), nullable=False, default="Asia/Manila")
    # e.g. "NAMRIA", "CSV_MANUAL".
    data_source: Mapped[str] = mapped_column(String(64), nullable=False)
    # Vertical datum this station's heights are referenced to (e.g. "MLWS",
    # "MLLW", "LAT"). Kept per-station; datums are not interchangeable.
    vertical_datum: Mapped[str | None] = mapped_column(String(64))
    measurement_unit: Mapped[str] = mapped_column(String(16), nullable=False, default="meters")

    # Supported prediction period (inclusive), stored in UTC.
    prediction_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    prediction_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    country: Mapped[Country] = relationship(back_populates="tide_stations")
    monitoring_station: Mapped[MonitoringStation | None] = relationship(
        back_populates="tide_stations"
    )
    datasets: Mapped[list[TideDataset]] = relationship(back_populates="station")
    predictions: Mapped[list[TidePrediction]] = relationship(back_populates="station")

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<TideStation {self.code}>"


class TideDataset(Base, TimestampMixin):
    """An imported, immutable source dataset for a tide station."""

    __tablename__ = "tide_datasets"

    id: Mapped[int] = mapped_column(primary_key=True)
    tide_station_id: Mapped[int] = mapped_column(
        ForeignKey("tide_stations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    source: Mapped[str] = mapped_column(String(64), nullable=False)  # e.g. "NAMRIA_CSV"
    description: Mapped[str | None] = mapped_column(Text)

    unit: Mapped[str] = mapped_column(String(16), nullable=False, default="meters")
    vertical_datum: Mapped[str | None] = mapped_column(String(64))

    coverage_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    coverage_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    record_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Provenance of the raw file (never commit real datasets to git).
    source_filename: Mapped[str | None] = mapped_column(String(300))
    file_hash: Mapped[str | None] = mapped_column(String(128), index=True)

    station: Mapped[TideStation] = relationship(back_populates="datasets")
    predictions: Mapped[list[TidePrediction]] = relationship(back_populates="dataset")
    imports: Mapped[list[DatasetImport]] = relationship(back_populates="dataset")

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<TideDataset {self.id} {self.name!r}>"


class TidePrediction(Base):
    """A single predicted tide height at a timestamp.

    TimescaleDB hypertable partitioned on ``timestamp``. The composite primary
    key ``(tide_station_id, timestamp)`` deliberately includes the time column,
    as required by TimescaleDB.
    """

    __tablename__ = "tide_predictions"
    __table_args__ = (
        Index("ix_tide_predictions_station_ts", "tide_station_id", "timestamp"),
        {"comment": "TimescaleDB hypertable on timestamp"},
    )

    tide_station_id: Mapped[int] = mapped_column(
        ForeignKey("tide_stations.id", ondelete="CASCADE"), primary_key=True
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, nullable=False
    )
    tide_dataset_id: Mapped[int | None] = mapped_column(
        ForeignKey("tide_datasets.id", ondelete="SET NULL"), nullable=True, index=True
    )

    # Predicted height (never a measured sea level).
    height: Mapped[float] = mapped_column(Float, nullable=False)
    height_unit: Mapped[str] = mapped_column(String(16), nullable=False, default="meters")
    vertical_datum: Mapped[str | None] = mapped_column(String(64))
    # "high" / "low" where known, else NULL.
    event_type: Mapped[str | None] = mapped_column(String(16))
    # data quality token, e.g. "ok", "interpolated", "unknown".
    quality_status: Mapped[str] = mapped_column(String(32), nullable=False, default="unknown")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    station: Mapped[TideStation] = relationship(back_populates="predictions")
    dataset: Mapped[TideDataset | None] = relationship(back_populates="predictions")


class DatasetImport(Base, TimestampMixin):
    """Insert-only import history for CSV / dataset ingestion.

    Nothing is ever discarded silently: rejected records and their reasons are
    recorded here (and, where possible, in a companion rejected-records store).
    """

    __tablename__ = "dataset_imports"
    __table_args__ = (Index("ix_dataset_imports_station_created", "tide_station_id", "created_at"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    tide_station_id: Mapped[int] = mapped_column(
        ForeignKey("tide_stations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    tide_dataset_id: Mapped[int | None] = mapped_column(
        ForeignKey("tide_datasets.id", ondelete="SET NULL"), nullable=True, index=True
    )

    source_filename: Mapped[str | None] = mapped_column(String(300))
    file_hash: Mapped[str | None] = mapped_column(String(128))
    # Configurable column mapping used during the import.
    column_mapping: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    total_records: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    imported_records: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rejected_records: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # e.g. {"invalid_date": 3, "duplicate": 1, "missing_height": 2}
    rejection_summary: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    message: Mapped[str | None] = mapped_column(Text)

    station: Mapped[TideStation] = relationship()
    dataset: Mapped[TideDataset | None] = relationship(back_populates="imports")

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<DatasetImport {self.id} status={self.status}>"
