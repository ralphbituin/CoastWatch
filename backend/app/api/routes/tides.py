"""Tide routes: stations, CSV import, predictions and phase."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_tide_provider, pagination_params
from app.core.config import settings
from app.core.exceptions import ConflictError, NotFoundError, ValidationError
from app.models.tide import DatasetImport, TideDataset, TideStation
from app.schemas.tides import (
    DatasetImportDetail,
    DatasetImportRead,
    ImportCommitResponse,
    ImportListResponse,
    ImportPreviewResponse,
    TideChartResponse,
    TideEventRead,
    TidePhaseResponse,
    TidePointRead,
    TideStationCreate,
    TideStationRead,
    quality_note,
)
from app.services.tides.base import HighLowWater
from app.services.tides.interpolation import InterpolationMethod
from app.services.tides.namria import NAMRIATideProvider

router = APIRouter(tags=["tides"])

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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


def _event_read(event: HighLowWater) -> TideEventRead:
    return TideEventRead(
        timestamp=event.timestamp,
        height=event.height,
        type=event.event.value,
        unit=event.unit,
        datum=event.datum,
    )


def _local_iso(moment: datetime, timezone: str) -> str:
    return moment.astimezone(ZoneInfo(timezone)).isoformat()


def _phase_response(
    *,
    station_code: str,
    at: datetime,
    method: InterpolationMethod,
    provider: NAMRIATideProvider,
) -> TidePhaseResponse:
    result = provider.get_prediction_at(station_code, at, method=method)
    station = provider.get_station(station_code)

    previous = _event_read(result.previous_event) if result.previous_event else None
    upcoming = _event_read(result.next_event) if result.next_event else None

    return TidePhaseResponse(
        station_code=station.code,
        station_name=station.name,
        requested_timestamp=result.timestamp,
        local_timestamp=_local_iso(result.timestamp, station.timezone),
        server_time=datetime.now(UTC),
        predicted_height_m=round(result.height, 4),
        unit=result.unit,
        vertical_datum=result.datum,
        direction=result.direction.value,
        quality=result.quality.value,
        quality_note=quality_note(result.quality.value),
        source=result.source,
        dataset_id=result.dataset_id,
        timezone=station.timezone,
        interpolation_method=method.value,
        previous_event=previous,
        next_event=upcoming,
    )


def _parse_mapping(raw: str | None) -> dict | None:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValidationError(
            "column_mapping must be valid JSON", details={"error": str(exc)}
        ) from exc
    if not isinstance(parsed, dict):
        raise ValidationError("column_mapping must be a JSON object")
    return parsed


# ---------------------------------------------------------------------------
# Stations
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------
@router.get(
    "/predictions",
    response_model=list[TidePointRead],
    summary="Predictions for a time range",
)
def list_predictions(
    station_code: str = Query(...),
    start: datetime = Query(...),
    end: datetime = Query(...),
    provider: NAMRIATideProvider = Depends(get_tide_provider),
) -> list[TidePointRead]:
    if end <= start:
        raise ValidationError("end must be after start")
    points = provider.get_predictions(station_code, start, end)
    return [
        TidePointRead(
            timestamp=p.timestamp,
            height=p.height,
            unit=p.unit,
            datum=p.datum,
            quality_status=p.quality.value,
        )
        for p in points
    ]


@router.get(
    "/high-low-waters",
    response_model=list[TideEventRead],
    summary="High and low water events for a range",
)
def high_low_waters(
    station_code: str = Query(...),
    start: datetime = Query(...),
    end: datetime = Query(...),
    provider: NAMRIATideProvider = Depends(get_tide_provider),
) -> list[TideEventRead]:
    if end <= start:
        raise ValidationError("end must be after start")
    events = provider.get_high_low_waters(station_code, start, end)
    return [_event_read(e) for e in events]


@router.get(
    "/current",
    response_model=TidePhaseResponse,
    summary="CURRENT TIME: predicted tide at the current timestamp",
)
def current_tide(
    station_code: str = Query(...),
    method: InterpolationMethod = Query(default=InterpolationMethod.PCHIP),
    provider: NAMRIATideProvider = Depends(get_tide_provider),
) -> TidePhaseResponse:
    at = datetime.now(UTC)
    return _phase_response(station_code=station_code, at=at, method=method, provider=provider)


@router.get(
    "/at",
    response_model=TidePhaseResponse,
    summary="SELECTED TIME: predicted tide at a chosen timestamp",
)
def tide_at(
    station_code: str = Query(...),
    timestamp: datetime = Query(...),
    method: InterpolationMethod = Query(default=InterpolationMethod.PCHIP),
    provider: NAMRIATideProvider = Depends(get_tide_provider),
) -> TidePhaseResponse:
    station = provider.get_station(station_code)
    at = timestamp
    if at.tzinfo is None:
        # Naive input is interpreted in the station's local time zone.
        at = at.replace(tzinfo=ZoneInfo(station.timezone))
    return _phase_response(station_code=station_code, at=at, method=method, provider=provider)


@router.get(
    "/chart",
    response_model=TideChartResponse,
    summary="Chart data (24h/48h/7d) plus high/low water events",
)
def chart(
    station_code: str = Query(...),
    start: datetime | None = Query(default=None),
    end: datetime | None = Query(default=None),
    method: InterpolationMethod = Query(default=InterpolationMethod.PCHIP),
    provider: NAMRIATideProvider = Depends(get_tide_provider),
) -> TideChartResponse:
    station = provider.get_station(station_code)
    coverage_start, coverage_end = provider.station_coverage(station)

    if end is None:
        end = datetime.now(UTC)
    if start is None:
        start = end - timedelta(hours=24)
    if end <= start:
        raise ValidationError("end must be after start")

    points = provider.get_predictions(station_code, start, end)
    events = provider.get_high_low_waters(station_code, start, end)

    sufficient_for_7d = (
        coverage_start is not None
        and coverage_end is not None
        and (coverage_end - coverage_start) >= timedelta(days=7)
    )
    return TideChartResponse(
        station_code=station_code,
        start=start,
        end=end,
        method=method.value,
        count=len(points),
        coverage_start=coverage_start,
        coverage_end=coverage_end,
        sufficient_for_7d=sufficient_for_7d,
        points=[
            TidePointRead(
                timestamp=p.timestamp,
                height=p.height,
                unit=p.unit,
                datum=p.datum,
                quality_status=p.quality.value,
            )
            for p in points
        ],
        events=[_event_read(e) for e in events],
    )


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
@router.post(
    "/imports/preview",
    response_model=ImportPreviewResponse,
    summary="Preview a CSV import (no database changes)",
)
async def preview_import(
    file: UploadFile = File(...),
    timezone: str = Form(settings.default_timezone),
    column_mapping: str | None = Form(default=None),
    provider: NAMRIATideProvider = Depends(get_tide_provider),
) -> ImportPreviewResponse:
    content = await _read_upload(file)
    mapping = _parse_mapping(column_mapping)
    result = provider.preview(content, mapping=mapping, timezone=timezone)
    result.pop("station_code", None)
    return ImportPreviewResponse(station_code=None, **result)


@router.post(
    "/imports",
    response_model=ImportCommitResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Import a NAMRIA-style CSV into a tide station",
)
async def commit_import(
    file: UploadFile = File(...),
    station_code: str = Form(...),
    name: str = Form(...),
    timezone: str = Form(settings.default_timezone),
    unit: str = Form("meters"),
    vertical_datum: str | None = Form(default=None),
    description: str | None = Form(default=None),
    source: str = Form("NAMRIA_CSV"),
    column_mapping: str | None = Form(default=None),
    provider: NAMRIATideProvider = Depends(get_tide_provider),
) -> ImportCommitResponse:
    content = await _read_upload(file)
    mapping = _parse_mapping(column_mapping)
    result = provider.import_data(
        content,
        station_code=station_code,
        name=name,
        source=source,
        unit=unit.lower(),
        vertical_datum=vertical_datum,
        timezone=timezone,
        description=description,
        column_mapping=mapping,
        source_filename=file.filename,
        commit=True,
    )
    return ImportCommitResponse(**result)


@router.get(
    "/imports",
    response_model=ImportListResponse,
    summary="Import history (latest first)",
)
def list_imports(
    station_code: str | None = Query(default=None),
    page: tuple[int, int] = Depends(pagination_params),
    db: Session = Depends(get_db),
) -> ImportListResponse:
    limit, offset = page
    query = select(DatasetImport).order_by(DatasetImport.created_at.desc())
    if station_code:
        station = db.scalar(select(TideStation).where(TideStation.code == station_code))
        if station is None:
            raise NotFoundError(f"Tide station not found: {station_code}")
        query = query.where(DatasetImport.tide_station_id == station.id)

    total = db.scalar(select(func.count()).select_from(query.subquery())) or 0
    imports = db.scalars(query.limit(limit).offset(offset)).all()
    stations = _station_codes(db, [i.tide_station_id for i in imports])
    return ImportListResponse(
        imports=[_import_read(i, stations.get(i.tide_station_id, "??")) for i in imports],
        total=total,
    )


@router.get(
    "/imports/{import_id}",
    response_model=DatasetImportDetail,
    summary="Import detail, including rejected records and reasons",
)
def get_import(import_id: int, db: Session = Depends(get_db)) -> DatasetImportDetail:
    imp = db.get(DatasetImport, import_id)
    if imp is None:
        raise NotFoundError(f"Dataset import not found: {import_id}")
    station_code = (
        db.scalar(select(TideStation.code).where(TideStation.id == imp.tide_station_id)) or "??"
    )
    return DatasetImportDetail(
        **_import_read(imp, station_code).model_dump(),
        column_mapping=imp.column_mapping,
        rejection_summary=imp.rejection_summary,
        rejection_samples=imp.rejection_samples,
        message=imp.message,
    )


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _country_by_iso(db: Session, iso: str):
    from app.models.country import Country

    country = db.scalar(select(Country).where(Country.iso_code == iso.upper()))
    if country is None:
        raise ValidationError(
            f"Country {iso!r} is not known. Seed it first or use an existing iso_code.",
            details={"iso_code": iso},
        )
    return country


async def _read_upload(file: UploadFile) -> bytes:
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValidationError(
            f"Upload too large ({len(content)} bytes; max {MAX_UPLOAD_BYTES}).",
            details={"max_bytes": MAX_UPLOAD_BYTES},
        )
    if not content.strip():
        raise ValidationError("Uploaded file is empty.")
    return content


def _station_codes(db: Session, ids: list[int]) -> dict[int, str]:
    if not ids:
        return {}
    rows = db.execute(select(TideStation.id, TideStation.code).where(TideStation.id.in_(ids))).all()
    return {row[0]: row[1] for row in rows}


def _import_read(imp: DatasetImport, station_code: str) -> DatasetImportRead:
    return DatasetImportRead(
        id=imp.id,
        tide_station_id=imp.tide_station_id,
        station_code=station_code,
        dataset_id=imp.tide_dataset_id,
        source_filename=imp.source_filename,
        file_hash=imp.file_hash,
        total_records=imp.total_records,
        imported_records=imp.imported_records,
        rejected_records=imp.rejected_records,
        status=imp.status,
        created_at=imp.created_at,
    )
