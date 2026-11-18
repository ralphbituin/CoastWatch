"""Monitoring-station routes: stations, tide-station associations."""

from __future__ import annotations

from fastapi import APIRouter, Depends, status
from geoalchemy2.elements import WKTElement
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.core.exceptions import ConflictError, NotFoundError, ValidationError
from app.models.country import Country
from app.models.station import MonitoringStation
from app.models.tide import TideStation
from app.schemas.common import Message
from app.schemas.stations import MonitoringStationCreate, MonitoringStationRead
from app.schemas.tides import TideStationRead

router = APIRouter(tags=["stations"])


def _station_counts(db: Session, station_ids: list[int]) -> dict[int, int]:
    if not station_ids:
        return {}
    rows = db.execute(
        select(TideStation.monitoring_station_id, func.count(TideStation.id))
        .where(TideStation.monitoring_station_id.in_(station_ids))
        .group_by(TideStation.monitoring_station_id)
    ).all()
    return {row[0]: row[1] for row in rows}


def _to_read(st: MonitoringStation, tide_station_count: int) -> MonitoringStationRead:
    return MonitoringStationRead(
        id=st.id,
        code=st.code,
        name=st.name,
        region=st.region,
        country_iso=st.country.iso_code if st.country else "??",
        country_name=st.country.name if st.country else "Unknown",
        latitude=st.latitude,
        longitude=st.longitude,
        timezone=st.timezone,
        vertical_datum=st.vertical_datum,
        description=st.description,
        tide_station_count=tide_station_count,
        created_at=st.created_at,
    )


def _get_station(db: Session, station_id: int) -> MonitoringStation:
    station = db.get(MonitoringStation, station_id)
    if station is None:
        raise NotFoundError(f"Monitoring station not found: {station_id}")
    return station


def _country_by_iso(db: Session, iso: str) -> Country:
    country = db.scalar(select(Country).where(Country.iso_code == iso.upper()))
    if country is None:
        raise ValidationError(
            f"Country {iso!r} is not known. Seed it first or use an existing iso_code.",
            details={"iso_code": iso},
        )
    return country


def _tide_station_read(st: TideStation) -> TideStationRead:
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
        dataset_count=0,  # filled below when listed per station context
    )


@router.get("", response_model=list[MonitoringStationRead], summary="List monitoring stations")
def list_stations(db: Session = Depends(get_db)) -> list[MonitoringStationRead]:
    stations = db.scalars(select(MonitoringStation).order_by(MonitoringStation.code)).all()
    counts = _station_counts(db, [s.id for s in stations])
    return [_to_read(s, counts.get(s.id, 0)) for s in stations]


@router.post(
    "",
    response_model=MonitoringStationRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create a monitoring station",
)
def create_station(
    payload: MonitoringStationCreate, db: Session = Depends(get_db)
) -> MonitoringStationRead:
    country = _country_by_iso(db, payload.country_iso)
    station = MonitoringStation(
        country_id=country.id,
        code=payload.code,
        name=payload.name,
        region=payload.region,
        latitude=payload.latitude,
        longitude=payload.longitude,
        location=WKTElement(f"POINT({payload.longitude} {payload.latitude})", srid=4326),
        timezone=payload.timezone,
        vertical_datum=payload.vertical_datum,
        description=payload.description,
    )
    db.add(station)
    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        raise ConflictError(
            f"Monitoring station code already exists or constraint failed: {payload.code}",
            details={"code": payload.code},
        ) from exc
    db.refresh(station)
    return _to_read(station, 0)


@router.get(
    "/{station_id}", response_model=MonitoringStationRead, summary="Monitoring station detail"
)
def get_station(station_id: int, db: Session = Depends(get_db)) -> MonitoringStationRead:
    station = _get_station(db, station_id)
    counts = _station_counts(db, [station.id])
    return _to_read(station, counts.get(station.id, 0))


@router.get(
    "/{station_id}/tide-stations",
    response_model=list[TideStationRead],
    summary="Tide stations associated with this monitoring station",
)
def list_linked_tide_stations(
    station_id: int,
    db: Session = Depends(get_db),
) -> list[TideStationRead]:
    _get_station(db, station_id)  # 404 if unknown
    stations = db.scalars(
        select(TideStation)
        .where(TideStation.monitoring_station_id == station_id)
        .order_by(TideStation.code)
    ).all()
    return [_tide_station_read(s) for s in stations]


@router.post(
    "/{station_id}/tide-stations/{tide_station_id}",
    response_model=TideStationRead,
    summary="Associate a tide station with a monitoring station",
)
def link_tide_station(
    station_id: int,
    tide_station_id: int,
    db: Session = Depends(get_db),
) -> TideStationRead:
    _get_station(db, station_id)
    tide_station = db.get(TideStation, tide_station_id)
    if tide_station is None:
        raise NotFoundError(f"Tide station not found: {tide_station_id}")
    tide_station.monitoring_station_id = station_id
    db.commit()
    db.refresh(tide_station)
    return _tide_station_read(tide_station)


@router.delete(
    "/{station_id}/tide-stations/{tide_station_id}",
    response_model=Message,
    summary="Disassociate a tide station from a monitoring station",
)
def unlink_tide_station(
    station_id: int,
    tide_station_id: int,
    db: Session = Depends(get_db),
) -> Message:
    _get_station(db, station_id)
    tide_station = db.get(TideStation, tide_station_id)
    if tide_station is None:
        raise NotFoundError(f"Tide station not found: {tide_station_id}")
    if tide_station.monitoring_station_id == station_id:
        tide_station.monitoring_station_id = None
        db.commit()
    # Idempotent: disassociating a station that is already unlinked is a no-op.
    return Message(message="tide station disassociated")
