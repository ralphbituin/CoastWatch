"""Shoreline routes (observations, calibrations). Implemented in Phase 4.

Until a validated segmentation model exists, waterlines come from
**operator-labelled masks** (`POST /shorelines/observations`) or, once a
model is trained and provisioned, from `POST /shorelines/observe`. Nothing
here fabricates a waterline: unavailable segmentation reports
``DATA_UNRELIABLE`` with an explicit reason.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app.api.deps import get_db, pagination_params
from app.core.exceptions import (
    DataUnavailableError,
    NotFoundError,
    ValidationError,
)
from app.core.logging import get_logger
from app.models.camera import Camera, CameraCalibration
from app.models.shoreline import ShorelineObservation
from app.models.station import MonitoringStation
from app.schemas.shorelines import (
    ShorelineCalibrationRead,
    ShorelineModuleStatus,
    ShorelineObservationDetail,
    ShorelineObservationSummary,
    ShorelineObserveRequest,
    ShorelineObserveResult,
)
from app.services.pipeline import CoastWatchPipeline, record_observation
from app.services.vision.calibration import (
    CalibrationProfile,
)
from app.services.vision.segmentation import find_model_artifact
from app.services.vision.shoreline import (
    ShorelineExtractionError,
    extract_shoreline,
)

logger = get_logger(__name__)

router = APIRouter(tags=["shorelines"])

MAX_MASK_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


def _get_camera(db: Session, camera_id: int) -> Camera:
    camera = db.get(Camera, camera_id, options=[joinedload(Camera.station)])
    if camera is None:
        raise NotFoundError(f"Camera not found: {camera_id}")
    return camera


def _observation_summary(
    row: ShorelineObservation,
    camera_code: str | None = None,
    station_code: str | None = None,
) -> ShorelineObservationSummary:
    return ShorelineObservationSummary(
        camera_id=row.camera_id,
        camera_code=camera_code,
        observed_at=row.observed_at,
        monitoring_station_id=row.monitoring_station_id,
        station_code=station_code,
        calibration_id=row.calibration_id,
        calibrated_position_m=row.calibrated_position_m,
        segmentation_confidence=row.segmentation_confidence,
        frame_quality_score=row.frame_quality_score,
        frame_quality_flags=list(row.frame_quality_flags or []),
        created_at=row.created_at,
    )


def _observation_detail(
    row: ShorelineObservation,
    camera_code: str | None = None,
    station_code: str | None = None,
) -> ShorelineObservationDetail:
    return ShorelineObservationDetail(
        camera_id=row.camera_id,
        camera_code=camera_code,
        observed_at=row.observed_at,
        monitoring_station_id=row.monitoring_station_id,
        station_code=station_code,
        calibration_id=row.calibration_id,
        shoreline_geometry=row.shoreline_geometry,
        calibrated_position_m=row.calibrated_position_m,
        segmentation_confidence=row.segmentation_confidence,
        frame_quality_score=row.frame_quality_score,
        frame_quality_flags=list(row.frame_quality_flags or []),
        source_frame_path=row.source_frame_path,
        created_at=row.created_at,
    )


@router.get("", response_model=ShorelineModuleStatus, summary="Shorelines module status")
def shorelines_status() -> ShorelineModuleStatus:
    available = find_model_artifact() is not None
    return ShorelineModuleStatus(
        segmentation_model_available=available,
        implemented_endpoints=[
            "GET /shorelines/observations",
            "GET /shorelines/observations/latest",
            "POST /shorelines/observations (operator-labelled mask)",
            "POST /shorelines/observe",
            "GET /shorelines/calibrations",
        ],
        notes=[
            "No segmentation model is provisioned yet; "
            "waterlines come from operator-labelled masks."
            if not available
            else "Segmentation artifact detected, but inference wiring is not complete.",
            "Baselines land in Phase 5; anomalies in Phase 6.",
        ],
    )


@router.get(
    "/observations",
    response_model=list[ShorelineObservationSummary],
    summary="List shoreline observations",
)
def list_observations(
    db: Session = Depends(get_db),
    camera_id: int | None = None,
    station_id: int | None = None,
    pagination: tuple[int, int] = Depends(pagination_params),
) -> list[ShorelineObservationSummary]:
    limit, offset = pagination
    stmt = (
        select(ShorelineObservation, Camera.code, MonitoringStation.code)
        .join(Camera, Camera.id == ShorelineObservation.camera_id)
        .join(MonitoringStation, MonitoringStation.id == ShorelineObservation.monitoring_station_id)
        .order_by(ShorelineObservation.observed_at.desc())
        .limit(limit)
        .offset(offset)
    )
    if camera_id is not None:
        stmt = stmt.where(ShorelineObservation.camera_id == camera_id)
    if station_id is not None:
        stmt = stmt.where(ShorelineObservation.monitoring_station_id == station_id)
    return [
        _observation_summary(row, camera_code, station_code)
        for row, camera_code, station_code in db.execute(stmt)
    ]


@router.get(
    "/observations/latest",
    response_model=ShorelineObservationDetail,
    summary="Latest shoreline observation (optionally per camera or station)",
)
def latest_observation(
    db: Session = Depends(get_db),
    camera_id: int | None = None,
    station_id: int | None = None,
) -> ShorelineObservationDetail:
    stmt = (
        select(ShorelineObservation, Camera.code, MonitoringStation.code)
        .join(Camera, Camera.id == ShorelineObservation.camera_id)
        .join(MonitoringStation, MonitoringStation.id == ShorelineObservation.monitoring_station_id)
        .order_by(ShorelineObservation.observed_at.desc())
        .limit(1)
    )
    if camera_id is not None:
        stmt = stmt.where(ShorelineObservation.camera_id == camera_id)
    if station_id is not None:
        stmt = stmt.where(ShorelineObservation.monitoring_station_id == station_id)
    row = db.execute(stmt).first()
    if row is None:
        raise DataUnavailableError(
            "No shoreline observations recorded yet",
            details={"camera_id": camera_id, "station_id": station_id},
        )
    observation, camera_code, station_code = row
    return _observation_detail(observation, camera_code, station_code)


@router.post(
    "/observations",
    response_model=ShorelineObservationDetail,
    status_code=status.HTTP_201_CREATED,
    summary="Record an observation from an operator-labelled water mask",
)
async def create_observation_from_mask(
    camera_id: int = Form(...),
    seaward_axis: str | None = Form(default=None),
    observed_at: str | None = Form(default=None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> ShorelineObservationDetail:
    camera = _get_camera(db, camera_id)
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg"}:
        raise ValidationError(f"Mask must be PNG/JPEG, got {suffix!r}")
    content = await file.read()
    if len(content) > MAX_MASK_UPLOAD_BYTES:
        raise ValidationError(
            f"Mask too large ({len(content)} bytes; max {MAX_MASK_UPLOAD_BYTES}).",
            details={"max_bytes": MAX_MASK_UPLOAD_BYTES},
        )
    mask = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValidationError("Mask file could not be decoded as an image")

    axis: tuple[float, float] | None = None
    if seaward_axis:
        try:
            parts = [float(p) for p in seaward_axis.split(",")]
            if len(parts) != 2:
                raise ValueError
            axis = (parts[0], parts[1])
        except ValueError as exc:
            raise ValidationError("seaward_axis must be 'dx,dy' (e.g. '0.94,0.34')") from exc

    parsed_observed_at: datetime | None = None
    if observed_at:
        try:
            parsed_observed_at = datetime.fromisoformat(observed_at)
        except ValueError as exc:
            raise ValidationError(
                "observed_at must be ISO 8601 (e.g. 2026-11-19T06:30:00+00:00)"
            ) from exc

    calibration = None
    if camera.active_calibration_id:
        row = db.get(CameraCalibration, camera.active_calibration_id)
        if row is not None and row.homography_matrix is not None:
            calibration = CalibrationProfile(
                camera_id=camera.id,
                version=row.version,
                homography=row.homography_matrix,
                meters_per_pixel=row.meters_per_pixel,
            )
    try:
        result = extract_shoreline(mask, calibration=calibration, seaward_axis=axis)
    except ShorelineExtractionError as exc:
        raise ValidationError(str(exc), details={"camera_id": camera.id}) from exc

    if axis is not None and calibration is None:
        # seaward_axis is only meaningful with a calibrated camera; surface that.
        result.calibrated_position_m = None

    observation = record_observation(
        db,
        camera,
        observed_at=parsed_observed_at or datetime.now(UTC),
        result=result,
        calibration_id=camera.active_calibration_id,
        quality_score=None,
        quality_flags=[],
        source_frame_path=None,
    )
    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.exception("failed to persist operator-mask observation for camera %s", camera.id)
        raise ValidationError(f"could not persist observation: {exc}") from exc
    db.refresh(observation)
    station_code = camera.station.code if camera.station else None
    return _observation_detail(observation, camera.code, station_code)


@router.post(
    "/observe",
    response_model=ShorelineObserveResult,
    summary="Run one observation cycle (camera → vision) for a camera",
)
def observe(
    payload: ShorelineObserveRequest, db: Session = Depends(get_db)
) -> ShorelineObserveResult:
    pipeline = CoastWatchPipeline(db)
    result = pipeline.process_once(station_id=payload.station_id, camera_id=payload.camera_id)
    return ShorelineObserveResult(
        station_id=result.station_id,
        camera_id=result.camera_id,
        observed_at=result.observed_at,
        actual_shoreline_m=result.actual_shoreline_m,
        expected_shoreline_m=result.expected_shoreline_m,
        residual_m=result.residual_m,
        retreat_velocity_m_s=result.retreat_velocity_m_s,
        anomaly_score=result.anomaly_score,
        state=result.state,  # type: ignore[arg-type]
        details=result.details,
    )


@router.get(
    "/calibrations",
    response_model=list[ShorelineCalibrationRead],
    summary="List calibration profiles across cameras (read-only)",
)
def list_calibrations(db: Session = Depends(get_db)) -> list[ShorelineCalibrationRead]:
    rows = db.execute(
        select(CameraCalibration, Camera.code)
        .join(Camera, Camera.id == CameraCalibration.camera_id)
        .order_by(Camera.code, CameraCalibration.version.desc())
    ).all()
    return [
        ShorelineCalibrationRead(
            id=cal.id,
            camera_id=cal.camera_id,
            camera_code=code,
            version=cal.version,
            meters_per_pixel=cal.meters_per_pixel,
            is_active=cal.is_active,
            valid_from=cal.valid_from,
            valid_to=cal.valid_to,
            notes=cal.notes,
            created_at=cal.created_at,
        )
        for cal, code in rows
    ]


@router.get(
    "/baselines",
    summary="Baselines land in Phase 5 (not implemented)",
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
)
def baselines_not_implemented() -> Any:
    return {
        "error": {
            "code": "not_implemented",
            "message": "Tide-aware shoreline baselines are implemented in Phase 5.",
            "details": {"phase": 5},
        }
    }
