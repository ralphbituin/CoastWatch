"""Tide-station routes.

Tide prediction generation and CSV import were removed from the product. The
station registry and its dataset metadata remain so existing tide data stays
inspectable and the tide tables are preserved for a future baseline.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.core.exceptions import ConflictError, NotFoundError, ValidationError
from app.models.country import Country
from app.models.tide import TideDataset, TideStation
from app.schemas.tides import TideStationCreate, TideStationRead

router = APIRouter(tags=["tides"])


def _dataset_counts(db: Session, station_ids: list[int]) -> dict[int, int]:
    if not station_ids:
        return {}
    rows = db.execute(
        select(TideDataset.tide_station_id, func.count(TideDataset.id))
        .where(TideDataset.tide_station_id.in_(station_ids))
        .group_by(TideDataset.tide_station_id)
    ).all()
    return {row[0]: row[1] for row in rows}


def _to_station_read(st: TideStation, dataset_count: int) -> TideStationRead:
    return TideStationRead(
        id=st.id,
        code=st.code,
        name=st.name,
        region=st.region,
        country_iso=st.country.iso_code if st.country else "??",
        country_name=st.country.name if st.country else "Unknown",
        latitude=st.latitude,
        longitude=st.longitude,
        timezone=st.timezone,
        data_source=st.data_source,
        vertical_datum=st.vertical_datum,
        measurement_unit=st.measurement_unit,
        prediction_start=st.prediction_start,
        prediction_end=st.prediction_end,
        monitoring_station_id=st.monitoring_station_id,
        dataset_count=dataset_count,
    )


def _country_by_iso(db: Session, iso: str) -> Country:
    country = db.scalar(select(Country).where(Country.iso_code == iso.upper()))
    if country is None:
        raise ValidationError(
            f"Country {iso!r} is not known. Seed it first or use an existing iso_code.",
            details={"iso_code": iso},
        )
    return country


@router.get("/stations", response_model=list[TideStationRead], summary="List tide stations")
def list_stations(db: Session = Depends(get_db)) -> list[TideStationRead]:
    stations = db.scalars(select(TideStation).order_by(TideStation.code)).all()
    counts = _dataset_counts(db, [s.id for s in stations])
    return [_to_station_read(s, counts.get(s.id, 0)) for s in stations]


@router.post(
    "/stations",
    response_model=TideStationRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create a tide station",
)
def create_station(payload: TideStationCreate, db: Session = Depends(get_db)) -> TideStationRead:
    from geoalchemy2.elements import WKTElement

    country = _country_by_iso(db, payload.country_iso)
    station = TideStation(
        country_id=country.id,
        code=payload.code,
        name=payload.name,
        region=payload.region,
        latitude=payload.latitude,
        longitude=payload.longitude,
        location=WKTElement(f"POINT({payload.longitude} {payload.latitude})", srid=4326),
        timezone=payload.timezone,
        data_source=payload.data_source,
        vertical_datum=payload.vertical_datum,
        measurement_unit=payload.measurement_unit,
        monitoring_station_id=payload.monitoring_station_id,
    )
    db.add(station)
    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        raise ConflictError(
            f"Tide station code already exists or constraint failed: {payload.code}",
            details={"code": payload.code},
        ) from exc
    db.refresh(station)
    return _to_station_read(station, 0)


@router.get("/stations/{code}", response_model=TideStationRead, summary="Tide station detail")
def get_station(code: str, db: Session = Depends(get_db)) -> TideStationRead:
    station = db.scalar(select(TideStation).where(TideStation.code == code))
    if station is None:
        raise NotFoundError(f"Tide station not found: {code}")
    counts = _dataset_counts(db, [station.id])
    return _to_station_read(station, counts.get(station.id, 0))
